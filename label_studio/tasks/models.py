"""This file and its contents are licensed under the Apache License 2.0. Please see the included NOTICE for copyright information and LICENSE for a copy of the license."""

import base64
import datetime
import logging
import numbers
import os
import random
import traceback
import uuid
from typing import Any, Mapping, Optional, Union, cast
from urllib.parse import urljoin

import ujson as json
from core.bulk_update_utils import bulk_update
from core.current_request import CurrentContext, get_current_request
from core.feature_flags import flag_set
from core.label_config import SINGLE_VALUED_TAGS
from core.redis import start_job_async_or_sync
from core.utils.common import (
    find_first_one_to_one_related_field_by_prefix,
    load_func,
    string_is_url,
    temporary_disconnect_list_signal,
)
from core.utils.db import batch_delete, fast_first
from core.utils.params import get_env
from data_import.models import FileUpload
from data_manager.managers import PreparedTaskManager, TaskManager
from django.conf import settings
from django.db import OperationalError, models, transaction
from django.db.models import CheckConstraint, F, JSONField, Q
from django.db.models.lookups import GreaterThanOrEqual
from django.db.models.signals import post_delete, post_save, pre_delete, pre_save
from django.dispatch import Signal, receiver
from django.urls import reverse
from django.utils.timesince import timesince
from django.utils.timezone import now
from django.utils.translation import gettext_lazy as _
from django.contrib.gis.db import models as gis
from django.contrib.postgres.indexes import GistIndex
from fsm.models import FsmHistoryStateModel
from fsm.queryset_mixins import FSMStateQuerySetMixin
from label_studio_sdk.label_interface.objects import PredictionValue
from rest_framework.exceptions import ValidationError
from tasks.choices import ActionType

logger = logging.getLogger(__name__)

TaskMixin = load_func(settings.TASK_MIXIN)


class Task(TaskMixin, FsmHistoryStateModel):
    """Business tasks from project"""

    id = models.AutoField(
        auto_created=True,
        primary_key=True,
        serialize=False,
        verbose_name='ID',
        db_index=True,
    )
    data = JSONField(
        'data',
        null=False,
        help_text='User imported or uploaded data for a task. Data is formatted according to '
        'the project label config. You can find examples of data for your project '
        'on the Import page in the Label Studio Data Manager UI.',
    )

    meta = JSONField(
        'meta',
        null=True,
        default=dict,
        help_text='Meta is user imported (uploaded) data and can be useful as input for an ML '
        'Backend for embeddings, advanced vectors, and other info. It is passed to '
        'ML during training/predicting steps.',
    )
    project = models.ForeignKey(
        'projects.Project',
        related_name='tasks',
        on_delete=models.CASCADE,
        null=True,
        help_text='Project ID for this task',
    )
    created_at = models.DateTimeField(_('created at'), auto_now_add=True, help_text='Time a task was created')
    updated_at = models.DateTimeField(_('updated at'), auto_now=True, help_text='Last time a task was updated')
    updated_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        related_name='updated_tasks',
        on_delete=models.SET_NULL,
        null=True,
        verbose_name=_('updated by'),
        help_text='Last annotator or reviewer who updated this task',
    )
    is_labeled = models.BooleanField(
        _('is_labeled'),
        default=False,
        help_text='True if the number of annotations for this task is greater than or equal '
        'to the number of maximum_completions for the project',
    )
    allow_skip = models.BooleanField(
        _('allow_skip'),
        default=True,
        null=True,
        help_text='Whether this task can be skipped. Set to False to make task unskippable.',
    )
    overlap = models.IntegerField(
        _('overlap'),
        default=1,
        db_index=True,
        help_text='Number of distinct annotators that processed the current task',
    )
    file_upload = models.ForeignKey(
        'data_import.FileUpload',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='tasks',
        help_text='Uploaded file used as data source for this task',
    )
    inner_id = models.BigIntegerField(
        _('inner id'),
        default=0,
        null=True,
        help_text='Internal task ID in the project, starts with 1',
    )
    updates = ['is_labeled']
    total_annotations = models.IntegerField(
        _('total_annotations'),
        default=0,
        db_index=True,
        help_text='Number of total annotations for the current task except cancelled annotations',
    )
    cancelled_annotations = models.IntegerField(
        _('cancelled_annotations'),
        default=0,
        db_index=True,
        help_text='Number of total cancelled annotations for the current task',
    )
    total_predictions = models.IntegerField(
        _('total_predictions'),
        default=0,
        db_index=True,
        help_text='Number of total predictions for the current task',
    )
    precomputed_agreement = models.FloatField(
        _('precomputed_agreement'),
        null=True,
        help_text='Average agreement score for the task',
    )

    comment_count = models.IntegerField(
        _('comment count'),
        default=0,
        db_index=True,
        help_text='Number of comments in the task including all annotations',
    )
    unresolved_comment_count = models.IntegerField(
        _('unresolved comment count'),
        default=0,
        db_index=True,
        help_text='Number of unresolved comments in the task including all annotations',
    )
    comment_authors = models.ManyToManyField(
        settings.AUTH_USER_MODEL,
        blank=True,
        default=None,
        related_name='tasks_with_comments',
        help_text='Users who wrote comments',
    )
    last_comment_updated_at = models.DateTimeField(
        _('last comment updated at'),
        default=None,
        null=True,
        db_index=True,
        help_text='When the last comment was updated',
    )
    # ✅ 추가: export 제외 플래그
    exclude_from_export = models.BooleanField(
        _('exclude from export'),
        default=False,
        db_index=True,   # export 필터링 성능을 위한 인덱스
        help_text='If true, this task will be excluded from exports',
    )

    objects = TaskManager()  # task manager by default
    prepared = PreparedTaskManager()  # task manager with filters, ordering, etc for data_manager app

    class Meta:
        db_table = 'task'
        indexes = [
            models.Index(fields=['project', 'is_labeled']),
            models.Index(fields=['project', 'inner_id']),
            models.Index(fields=['id', 'project']),
            models.Index(fields=['id', 'overlap']),
            models.Index(fields=['overlap']),
            models.Index(fields=['project', 'id']),
        ]

    @property
    def file_upload_name(self):
        return os.path.basename(self.file_upload.file.name)

    @classmethod
    def get_random(cls, project):
        """Get random task from a project, this should not be used lightly as its expensive method to run"""

        ids = cls.objects.filter(project=project).values_list('id', flat=True)
        if len(ids) == 0:
            return None

        random_id = random.choice(ids)

        return cls.objects.get(id=random_id)

    @classmethod
    def get_locked_by(cls, user, project=None, tasks=None):
        """Retrieve the task locked by specified user. Returns None if the specified user didn't lock anything."""
        lock = None
        if project is not None:
            lock = fast_first(TaskLock.objects.filter(user=user, expire_at__gt=now(), task__project=project))
        elif tasks is not None:
            locked_task = fast_first(tasks.filter(locks__user=user, locks__expire_at__gt=now()))
            if locked_task:
                return locked_task
        else:
            raise ValidationError('Neither project or tasks passed to get_locked_by')

        if lock:
            return lock.task

    def get_predictions_for_prelabeling(self):
        """This is called to return either new predictions from the
        model or grab static predictions if they were set, depending
        on the projects configuration.

        """
        from data_manager.functions import evaluate_predictions

        project = self.project
        predictions = self.predictions

        # TODO if we use live_model on project then we will need to check for it here
        if project.show_collab_predictions and project.model_version is not None:
            if project.ml_backend_in_model_version:
                new_predictions = evaluate_predictions([self])
                # TODO this is not as clean as I'd want it to
                # be. Effectively retrieve_predictions will work only for
                # tasks where there is no predictions matching current
                # model version. In case it will return a model_version
                # and we can grab predictions explicitly
                if isinstance(new_predictions, str):
                    model_version = new_predictions
                    return predictions.filter(model_version=model_version)
                else:
                    return new_predictions
            else:
                return predictions.filter(model_version=project.model_version)
        else:
            return []

    def get_lock_exclude_query(self, user):
        """
        Get query for excluding annotations from the lock check
        """
        SkipQueue = self.project.SkipQueue

        if self.project.skip_queue == SkipQueue.IGNORE_SKIPPED:
            # IGNORE_SKIPPED: my skipped tasks don't go anywhere
            # alien's and my skipped annotations are counted as regular annotations
            q = Q()
        else:
            if self.project.skip_queue == SkipQueue.REQUEUE_FOR_ME:
                # REQUEUE_FOR_ME means: only my skipped tasks go back to me,
                # alien's skipped annotations are counted as regular annotations
                q = Q(was_cancelled=True) & Q(completed_by=user)
            elif self.project.skip_queue == SkipQueue.REQUEUE_FOR_OTHERS:
                # REQUEUE_FOR_OTHERS: my skipped tasks go to others
                # alien's skipped annotations are not counted at all
                q = Q(was_cancelled=True) & ~Q(completed_by=user)
            else:
                raise ValidationError(f'Invalid SkipQueue value: {self.project.skip_queue}')

            # for LSE we also need to exclude rejected queue
            rejected_q = self.get_rejected_query()

            if rejected_q:
                q &= rejected_q

        return q | Q(ground_truth=True)

    def has_lock(self, user=None):
        """
        Check whether current task has been locked by some user

        Also has workaround for fixing not consistent is_labeled flag state
        """
        from projects.functions.next_task import get_next_task_logging_level

        if self.project.annotator_evaluation_enabled:
            # In annotator evaluation mode, ignore overlap setting for ground truth tasks
            if self.annotations.filter(ground_truth=True).exists():
                return False

        q = self.get_lock_exclude_query(user)

        num_locks = self.num_locks_user(user=user)
        num_annotations = self.annotations.exclude(q).count()
        num = num_locks + num_annotations

        if num > self.overlap_with_agreement_threshold(num, num_locks):
            logger.error(
                f'Num takes={num} > overlap={self.overlap} for task={self.id}, '
                f"skipped mode {self.project.skip_queue} - it's a bug",
                extra=dict(
                    lock_ttl=self.locks.values_list('user', 'expire_at'),
                    num_locks=num_locks,
                    num_annotations=num_annotations,
                ),
            )
            # TODO: remove this workaround after fixing the bug with inconsistent is_labeled flag
            if self.is_labeled is False:
                self.update_is_labeled()
                if self.is_labeled is True:
                    self.save(update_fields=['is_labeled'])

        result = bool(num >= self.overlap_with_agreement_threshold(num, num_locks))
        logger.log(
            get_next_task_logging_level(user),
            f'Task {self} locked: {result}; num_locks: {num_locks} num_annotations: {num_annotations} '
            f'skipped mode: {self.project.skip_queue}',
        )
        return result

    @property
    def num_locks(self):
        return self.locks.filter(expire_at__gt=now()).count()

    def overlap_with_agreement_threshold(self, num, num_locks):
        # Limit to one extra annotator at a time when the task is under the threshold and meets the overlap criteria,
        # regardless of the max_additional_annotators_assignable setting. This ensures recalculating agreement after
        # each annotation and prevents concurrent annotations from dropping the agreement below the threshold.
        if (
            hasattr(self.project, 'lse_project')
            and self.project.lse_project
            and self.project.lse_project.agreement_threshold is not None
        ):
            try:
                from stats.models import get_task_agreement
            except (ModuleNotFoundError, ImportError):
                return

            agreement = get_task_agreement(self)
            if agreement is not None and agreement < self.project.lse_project.agreement_threshold:
                return (
                    min(self.overlap + self.project.lse_project.max_additional_annotators_assignable, num + 1)
                    if num_locks == 0
                    else num
                )
        return self.overlap

    def num_locks_user(self, user):
        return self.locks.filter(expire_at__gt=now()).exclude(user=user).count()

    def get_storage_filename(self):
        for link_name in settings.IO_STORAGES_IMPORT_LINK_NAMES:
            if hasattr(self, link_name):
                return getattr(self, link_name).key

    def has_permission(self, user: 'User') -> bool:  # noqa: F821
        mixin_has_permission = cast(bool, super().has_permission(user))

        user.project = self.project  # link for activity log
        return mixin_has_permission and self.project.has_permission(user)

    def clear_expired_locks(self):
        self.locks.filter(expire_at__lt=now()).delete()

    def set_lock(self, user):
        """Lock current task by specified user. Lock lifetime is set by `expire_in_secs`"""
        from projects.functions.next_task import get_next_task_logging_level

        num_locks = self.num_locks
        if num_locks < self.overlap:
            lock_ttl = settings.TASK_LOCK_TTL
            if (
                flag_set('fflag_feat_all_leap_1534_custom_task_lock_timeout_short', user=user)
                and self.project.custom_task_lock_ttl
            ):
                lock_ttl = self.project.custom_task_lock_ttl
            expire_at = now() + datetime.timedelta(seconds=lock_ttl)
            try:
                task_lock = TaskLock.objects.get(task=self, user=user)
            except TaskLock.DoesNotExist:
                TaskLock.objects.create(task=self, user=user, expire_at=expire_at)
            else:
                task_lock.expire_at = expire_at
                task_lock.save()
            logger.log(
                get_next_task_logging_level(user),
                f'User={user} acquires a lock for the task={self} ttl: {lock_ttl}',
            )
        else:
            logger.error(
                f'Current number of locks for task {self.id} is {num_locks}, but overlap={self.overlap}: '
                f"that's a bug because this task should not be taken in a label stream (task should be locked)"
            )
        self.clear_expired_locks()

    def release_lock(self, user=None):
        """Release lock for the task.
        If user specified, it checks whether lock is released by the user who previously has locked that task
        """

        if user is not None:
            self.locks.filter(user=user).delete()
        else:
            self.locks.all().delete()
        self.clear_expired_locks()

    def get_storage_link(self):
        # TODO: how to get neatly any storage class here?
        return find_first_one_to_one_related_field_by_prefix(self, '.*io_storages_')

    @staticmethod
    def is_upload_file(filename):
        if not isinstance(filename, str):
            return False
        return filename.startswith(settings.UPLOAD_DIR + '/')

    @staticmethod
    def prepare_filename(filename):
        if isinstance(filename, str):
            return filename.replace(settings.MEDIA_URL, '')
        return filename

    def resolve_storage_uri(self, url) -> Optional[Mapping[str, Any]]:
        from io_storages.functions import get_storage_by_url

        # Instead of using self.storage, we check all storage objects for the project to
        # support imported tasks that point to another bucket
        storage_objects = self.project.get_all_import_storage_objects
        storage = get_storage_by_url(url, storage_objects)

        if storage:
            return {
                'url': storage.generate_http_url(url),
                'presign_ttl': storage.presign_ttl,
            }

    def resolve_uri(self, task_data, project):
        # DEPRECATED: use resolve_uris instead.
        from io_storages.functions import get_storage_by_url

        if project.task_data_login and project.task_data_password:
            protected_data = {}
            for key, value in task_data.items():
                if isinstance(value, str) and string_is_url(value):
                    path = (
                        reverse('projects-file-proxy', kwargs={'pk': project.pk})
                        + '?url='
                        + base64.urlsafe_b64encode(value.encode()).decode()
                    )
                    value = urljoin(settings.HOSTNAME, path)
                protected_data[key] = value
            return protected_data
        else:
            storage_objects = project.get_all_import_storage_objects

            # try resolve URLs via storage associated with that task
            for field in task_data:
                # file saved in django file storage
                prepared_filename = self.prepare_filename(task_data[field])
                if settings.CLOUD_FILE_STORAGE_ENABLED and self.is_upload_file(prepared_filename):
                    # permission check: resolve uploaded files to the project only
                    file_upload = fast_first(FileUpload.objects.filter(project=project, file=prepared_filename))
                    if file_upload is not None:
                        task_data[field] = file_upload.url
                    # it's very rare case, e.g. user tried to reimport exported file from another project
                    # or user wrote his django storage path manually
                    else:
                        task_data[field] = task_data[field] + '?not_uploaded_project_file'
                    continue

                # project storage
                # TODO: to resolve nested lists and dicts we should improve get_storage_by_url(),
                # Now always using get_storage_by_url to ensure the storage with the correct bucket is used
                # As a last fallback we can use self.storage which is the storage the Task was imported from
                storage = get_storage_by_url(task_data[field], storage_objects) or self.storage
                if storage:
                    try:
                        resolved_uri = storage.resolve_uri(task_data[field], self)
                    except Exception as exc:
                        logger.debug(exc, exc_info=True)
                        resolved_uri = None
                    if resolved_uri:
                        task_data[field] = resolved_uri
            return task_data

    def resolve_uris(self, task_data, project):
        """Resolve all cloud storage URIs across all project storages.

        Unlike resolve_uri which picks one storage per field and resolves
        only the first URI, this iterates all storages per field and resolves
        every URI each storage can handle.
        """
        if project.task_data_login and project.task_data_password:
            protected_data = {}
            for key, value in task_data.items():
                if isinstance(value, str) and string_is_url(value):
                    path = (
                        reverse('projects-file-proxy', kwargs={'pk': project.pk})
                        + '?url='
                        + base64.urlsafe_b64encode(value.encode()).decode()
                    )
                    value = urljoin(settings.HOSTNAME, path)
                protected_data[key] = value
            return protected_data

        storage_objects = project.get_all_import_storage_objects

        for field in task_data:
            prepared_filename = self.prepare_filename(task_data[field])
            if settings.CLOUD_FILE_STORAGE_ENABLED and self.is_upload_file(prepared_filename):
                file_upload = fast_first(FileUpload.objects.filter(project=project, file=prepared_filename))
                if file_upload is not None:
                    task_data[field] = file_upload.url
                else:
                    task_data[field] = task_data[field] + '?not_uploaded_project_file'
                continue

            for storage_object in storage_objects:
                try:
                    resolved = storage_object.resolve_uris(task_data[field], self)
                except Exception as exc:
                    logger.debug(exc, exc_info=True)
                    resolved = None
                if resolved:
                    task_data[field] = resolved

            fallback_storage = self.storage
            if fallback_storage:
                storage_ids = {s.id for s in storage_objects}
                if fallback_storage.id not in storage_ids:
                    try:
                        resolved = fallback_storage.resolve_uris(task_data[field], self)
                    except Exception as exc:
                        logger.debug(exc, exc_info=True)
                        resolved = None
                    if resolved:
                        task_data[field] = resolved

        return task_data

    @property
    def storage(self):
        # maybe task has storage link
        # TODO: this is bad idea to use storage from storage_link,
        # because storage resolver will be limited to this storage,
        # however, we may need to use other storages to resolve the uri.
        storage_link = self.get_storage_link()
        if storage_link:
            return storage_link.storage

        # or try global storage settings (only s3 for now)
        elif get_env('USE_DEFAULT_S3_STORAGE', default=False, is_bool=True):
            # TODO: this is used to access global environment storage settings.
            # We may use more than one and non-default S3 storage (like GCS, Azure)
            from io_storages.s3.models import S3ImportStorage

            return S3ImportStorage()

    @property
    def completed_annotations(self):
        """Annotations that we take into account when set completed status to the task"""
        if self.project.skip_queue == self.project.SkipQueue.IGNORE_SKIPPED:
            return self.annotations
        else:
            return self.annotations.filter(Q_finished_annotations)

    def increase_project_summary_counters(self):
        if hasattr(self.project, 'summary'):
            summary = self.project.summary
            summary.update_data_columns([self])

    def decrease_project_summary_counters(self):
        if hasattr(self.project, 'summary'):
            summary = self.project.summary
            summary.remove_data_columns([self])

    def ensure_unique_groundtruth(self, annotation_id):
        self.annotations.exclude(id=annotation_id).update(ground_truth=False)

    def save(self, *args, update_fields=None, **kwargs):
        if self.inner_id == 0:
            task = Task.objects.filter(project=self.project).order_by('-inner_id').first()
            max_inner_id = 1
            if task:
                max_inner_id = task.inner_id

            # max_inner_id might be None in the old projects
            self.inner_id = None if max_inner_id is None else (max_inner_id + 1)
            if update_fields is not None:
                update_fields = {'inner_id'}.union(update_fields)

        super().save(*args, update_fields=update_fields, **kwargs)

    @staticmethod
    def delete_tasks_without_signals(queryset):
        """
        Delete Tasks queryset with switched off signals in batches to minimize memory usage
        :param queryset: Tasks queryset
        """
        signals = [
            (post_delete, update_all_task_states_after_deleting_task, Task),
            (pre_delete, remove_data_columns, Task),
        ]
        with temporary_disconnect_list_signal(signals):
            return batch_delete(queryset, batch_size=500)

    @staticmethod
    def delete_tasks_without_signals_from_task_ids(task_ids):
        queryset = Task.objects.filter(id__in=task_ids)
        Task.delete_tasks_without_signals(queryset)

    def delete(self, *args, **kwargs):
        self.before_delete_actions()
        result = super().delete(*args, **kwargs)
        # set updated_at field of task to now()
        return result


pre_bulk_create = Signal()  # providing args 'objs' and 'batch_size'
post_bulk_create = Signal()  # providing args 'objs' and 'batch_size'


class AnnotationQuerySet(models.QuerySet):
    pass


class AnnotationQuerySetWithFSM(FSMStateQuerySetMixin, AnnotationQuerySet):
    pass


class AnnotationManager(models.Manager):
    """
    Manager for Annotation model with FSM state support.

    Provides:
    - User-scoped filtering
    - Bulk creation with signals
    - FSM state annotation support
    """

    def get_queryset(self):
        """Return AnnotationQuerySet with FSM state annotation support"""
        # Create a dynamic class that mixes FSM support into the queryset

        return AnnotationQuerySetWithFSM(self.model, using=self._db)

    def for_user(self, user):
        return self.get_queryset().filter(project__organization=user.active_organization)

    def with_state(self):
        """Return queryset with FSM state annotated."""
        return self.get_queryset().with_state()

    def bulk_create(self, objs, batch_size=None):
        pre_bulk_create.send(sender=self.model, objs=objs, batch_size=batch_size)
        res = super(AnnotationManager, self).bulk_create(objs, batch_size)
        post_bulk_create.send(sender=self.model, objs=objs, batch_size=batch_size)
        return res


AnnotationMixin = load_func(settings.ANNOTATION_MIXIN)


class Annotation(AnnotationMixin, FsmHistoryStateModel):
    """Annotations & Labeling results"""

    objects = AnnotationManager()

    result = JSONField(
        'result',
        null=True,
        default=None,
        help_text='The main value of annotator work - labeling result in JSON format',
    )

    task = models.ForeignKey(
        'tasks.Task',
        on_delete=models.CASCADE,
        related_name='annotations',
        null=True,
        help_text='Corresponding task for this annotation',
    )
    # duplicate relation to project for performance reasons
    project = models.ForeignKey(
        'projects.Project',
        related_name='annotations',
        on_delete=models.CASCADE,
        null=True,
        help_text='Project ID for this annotation',
    )
    completed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        related_name='annotations',
        on_delete=models.SET_NULL,
        null=True,
        help_text='User ID of the person who created this annotation',
    )
    updated_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        related_name='updated_annotations',
        on_delete=models.SET_NULL,
        null=True,
        verbose_name=_('updated by'),
        help_text='Last user who updated this annotation',
    )
    was_cancelled = models.BooleanField(_('was cancelled'), default=False, help_text='User skipped the task')
    ground_truth = models.BooleanField(
        _('ground_truth'),
        default=False,
        help_text='This annotation is a Ground Truth (ground_truth)',
    )
    created_at = models.DateTimeField(_('created at'), auto_now_add=True, help_text='Creation time')
    updated_at = models.DateTimeField(_('updated at'), auto_now=True, help_text='Last updated time')
    draft_created_at = models.DateTimeField(
        _('draft created at'), null=True, default=None, help_text='Draft creation time'
    )
    lead_time = models.FloatField(
        _('lead time'),
        null=True,
        default=None,
        help_text='How much time it took to annotate the task',
    )
    prediction = JSONField(
        _('prediction'),
        null=True,
        default=dict,
        help_text='Prediction viewed at the time of annotation',
    )
    result_count = models.IntegerField(_('result count'), default=0, help_text='Results inside of annotation counter')

    parent_prediction = models.ForeignKey(
        'tasks.Prediction',
        on_delete=models.SET_NULL,
        related_name='child_annotations',
        null=True,
        help_text='Points to the prediction from which this annotation was created',
    )
    parent_annotation = models.ForeignKey(
        'tasks.Annotation',
        on_delete=models.SET_NULL,
        related_name='child_annotations',
        null=True,
        help_text='Points to the parent annotation from which this annotation was created',
    )
    unique_id = models.UUIDField(default=uuid.uuid4, null=True, blank=True, unique=True, editable=False)
    import_id = models.BigIntegerField(
        default=None,
        null=True,
        blank=True,
        db_index=True,
        help_text="Original annotation ID that was at the import step or NULL if this annotation wasn't imported",
    )
    last_action = models.CharField(
        _('last action'),
        max_length=128,
        choices=ActionType.choices,
        help_text='Action which was performed in the last annotation history item',
        default=None,
        null=True,
    )
    last_created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        verbose_name=_('last created by'),
        help_text='User who created the last annotation history item',
        default=None,
        null=True,
    )
    bulk_created = models.BooleanField(
        _('bulk created'),
        default=False,
        db_default=False,
        null=True,
        help_text='Annotation was created in bulk mode',
    )

    class Meta:
        db_table = 'task_completion'
        indexes = [
            models.Index(fields=['created_at']),
            models.Index(fields=['ground_truth']),
            models.Index(fields=['id', 'task']),
            models.Index(fields=['last_action']),
            models.Index(fields=['project', 'ground_truth']),
            models.Index(fields=['project', 'id']),
            models.Index(fields=['project', 'was_cancelled']),
            models.Index(fields=['task', 'completed_by']),
            models.Index(fields=['task', 'ground_truth']),
            models.Index(fields=['task', 'was_cancelled']),
            models.Index(fields=['was_cancelled']),
        ]

    def created_ago(self):
        """Humanize date"""
        return timesince(self.created_at)

    def entities_num(self):
        res = self.result
        if isinstance(res, str):
            res = json.loads(res)
        if res is None:
            res = []

        return len(res)

    def has_permission(self, user: 'User') -> bool:  # noqa: F821
        mixin_has_permission = cast(bool, super().has_permission(user))

        user.project = self.project  # link for activity log
        return mixin_has_permission and self.project.has_permission(user)

    def increase_project_summary_counters(self):
        if hasattr(self.project, 'summary'):
            logger.debug(f'Increase project.summary counters from {self}')
            summary = self.project.summary
            summary.update_created_annotations_and_labels([self])

    def decrease_project_summary_counters(self):
        if hasattr(self.project, 'summary'):
            logger.debug(f'Decrease project.summary counters from {self}')
            summary = self.project.summary
            summary.remove_created_annotations_and_labels([self])

    def update_task(self):
        update_fields = ['updated_at']

        # updated_by
        request = get_current_request()
        if request:
            self.task.updated_by = request.user
            update_fields.append('updated_by')

        self.task.save(update_fields=update_fields, skip_fsm=True)

    def save(self, *args, update_fields=None, **kwargs):
        request = get_current_request()
        if request:
            self.updated_by = request.user
            if update_fields is not None:
                update_fields = {'updated_by'}.union(update_fields)

        unique_list = {result.get('id') for result in (self.result or [])}

        self.result_count = len(unique_list)
        if update_fields is not None:
            update_fields = {'result_count'}.union(update_fields)
        result = super().save(*args, update_fields=update_fields, **kwargs)

        self.update_task()
        return result

    def delete(self, *args, **kwargs):
        # Store task and project references before deletion

        result = super().delete(*args, **kwargs)
        self.update_task()
        self.on_delete_update_counters()

        return result

    def _update_task_state_after_deletion(self, task, project):
        """Update task FSM state after annotation deletion."""
        from fsm.functions import update_task_state_after_annotation_deletion

        update_task_state_after_annotation_deletion(task, project)

    def on_delete_update_counters(self):
        task = self.task
        project = self.project

        logger.debug(f'Start updating counters for task {task.id}.')
        if self.was_cancelled:
            cancelled = task.annotations.all().filter(was_cancelled=True).count()
            Task.objects.filter(id=task.id).update(cancelled_annotations=cancelled)
            logger.debug(f'On delete updated cancelled_annotations for task {task.id}')
        else:
            total = task.annotations.all().filter(was_cancelled=False).count()
            Task.objects.filter(id=task.id).update(total_annotations=total)
            logger.debug(f'On delete updated total_annotations for task {task.id}')

        logger.debug(f'Update task stats for task={task}')
        task.update_is_labeled()
        Task.objects.filter(id=task.id).update(is_labeled=task.is_labeled)

        # FSM: Update task state
        self._update_task_state_after_deletion(task, project)

        # remove annotation counters in project summary followed by deleting an annotation
        logger.debug('Remove annotation counters in project summary followed by deleting an annotation')
        self.decrease_project_summary_counters()


class TaskLockQuerySet(models.QuerySet):
    """Custom QuerySet for TaskLock model"""

    pass


class TaskLockQuerySetWithFSM(FSMStateQuerySetMixin, TaskLockQuerySet):
    pass


class TaskLockManager(models.Manager):
    """Manager for TaskLock with FSM state support"""

    def get_queryset(self):
        """Return QuerySet with FSM state annotation support"""
        return TaskLockQuerySetWithFSM(self.model, using=self._db)

    def with_state(self):
        """Return queryset with FSM state annotated."""
        return self.get_queryset().with_state()


class TaskLock(FsmHistoryStateModel):
    objects = TaskLockManager()
    task = models.ForeignKey(
        'tasks.Task',
        on_delete=models.CASCADE,
        related_name='locks',
        help_text='Locked task',
    )
    expire_at = models.DateTimeField(_('expire_at'))
    unique_id = models.UUIDField(default=uuid.uuid4, null=True, blank=True, unique=True, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        related_name='task_locks',
        on_delete=models.CASCADE,
        help_text='User who locked this task',
    )
    created_at = models.DateTimeField(_('created at'), auto_now_add=True, help_text='Creation time', null=True)

    def has_permission(self, user):
        return self.task.has_permission(user)


class AnnotationDraftQuerySet(models.QuerySet):
    """Custom QuerySet for AnnotationDraft model"""

    pass


class AnnotationDraftQuerySetWithFSM(FSMStateQuerySetMixin, AnnotationDraftQuerySet):
    pass


class AnnotationDraftManager(models.Manager):
    """Manager for AnnotationDraft with FSM state support"""

    def get_queryset(self):
        """Return QuerySet with FSM state annotation support"""
        return AnnotationDraftQuerySetWithFSM(self.model, using=self._db)

    def with_state(self):
        """Return queryset with FSM state annotated."""
        return self.get_queryset().with_state()


class AnnotationDraft(FsmHistoryStateModel):
    objects = AnnotationDraftManager()
    result = JSONField(_('result'), help_text='Draft result in JSON format')
    lead_time = models.FloatField(
        _('lead time'),
        blank=True,
        null=True,
        help_text='How much time it took to annotate the task',
    )
    task = models.ForeignKey(
        'tasks.Task',
        on_delete=models.CASCADE,
        related_name='drafts',
        blank=True,
        null=True,
        help_text='Corresponding task for this draft',
    )
    annotation = models.ForeignKey(
        'tasks.Annotation',
        on_delete=models.CASCADE,
        related_name='drafts',
        blank=True,
        null=True,
        help_text='Corresponding annotation for this draft',
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        related_name='drafts',
        on_delete=models.CASCADE,
        help_text='User who created this draft',
    )
    was_postponed = models.BooleanField(
        _('was postponed'),
        default=False,
        help_text='User postponed this draft (clicked a forward button) in the label stream',
        db_index=True,
    )
    import_id = models.BigIntegerField(
        default=None,
        null=True,
        blank=True,
        db_index=True,
        help_text="Original draft ID that was at the import step or NULL if this draft wasn't imported",
    )

    created_at = models.DateTimeField(_('created at'), auto_now_add=True, help_text='Creation time')
    updated_at = models.DateTimeField(_('updated at'), auto_now=True, help_text='Last update time')

    def created_ago(self):
        """Humanize date"""
        return timesince(self.created_at)

    def has_permission(self, user):
        user.project = self.task.project  # link for activity log
        return self.task.project.has_permission(user)

    def save(self, *args, **kwargs):
        with transaction.atomic():
            project = self.task.project
            # Lock projectsummary first to avoid deadlocks with annotation-reviews
            # which accesses projectsummary before annotationdraft
            if hasattr(project, 'summary'):
                from projects.models import ProjectSummary

                ProjectSummary.objects.select_for_update().filter(pk=project.summary.pk).first()
            super().save(*args, **kwargs)
            if hasattr(project, 'summary'):
                project.summary.update_created_labels_drafts([self])

    def delete(self, *args, **kwargs):
        with transaction.atomic():
            project = self.task.project
            # Lock projectsummary first to match save() and annotation-reviews lock ordering.
            # This prevents deadlocks with concurrent operations that access
            # projectsummary before annotationdraft.
            if hasattr(project, 'summary'):
                from projects.models import ProjectSummary

                ProjectSummary.objects.select_for_update().filter(pk=project.summary.pk).first()
            # Then lock annotationdraft row
            AnnotationDraft.objects.select_for_update().filter(pk=self.pk).first()
            if hasattr(project, 'summary'):
                project.summary.remove_created_drafts_and_labels([self])
            super().delete(*args, **kwargs)


class Prediction(models.Model):
    """ML backend / Prompts predictions"""

    result = JSONField('result', null=True, default=dict, help_text='Prediction result')
    score = models.FloatField(_('score'), default=None, help_text='Prediction score', null=True)
    model_version = models.TextField(
        _('model version'),
        default='',
        blank=True,
        null=True,
        help_text='A string value that for model version that produced the prediction. Used in both live models and when uploading offline predictions.',
    )

    model = models.ForeignKey(
        'ml.MLBackend',
        on_delete=models.CASCADE,
        related_name='predictions',
        null=True,
        help_text='An ML Backend instance that created the prediction.',
    )
    model_run = models.ForeignKey(
        'ml_models.ModelRun',
        on_delete=models.CASCADE,
        related_name='predictions',
        null=True,
        help_text='A run of a ModelVersion that created the prediction.',
    )
    cluster = models.IntegerField(
        _('cluster'),
        default=None,
        help_text='Cluster for the current prediction',
        null=True,
    )
    neighbors = JSONField(
        'neighbors',
        null=True,
        blank=True,
        help_text='Array of task IDs of the closest neighbors',
    )
    mislabeling = models.FloatField(_('mislabeling'), default=0.0, help_text='Related task mislabeling score')

    task = models.ForeignKey('tasks.Task', on_delete=models.CASCADE, related_name='predictions')
    project = models.ForeignKey('projects.Project', on_delete=models.CASCADE, related_name='predictions', null=True)
    created_at = models.DateTimeField(_('created at'), auto_now_add=True)
    updated_at = models.DateTimeField(_('updated at'), auto_now=True)

    def created_ago(self):
        """Humanize date"""
        return timesince(self.created_at)

    def has_permission(self, user):
        user.project = self.project  # link for activity log
        return self.project.has_permission(user)

    @classmethod
    def prepare_prediction_result(cls, result, project):
        """
        This function does the following logic of transforming "result" object:
        result is list -> use raw result as is
        result is dict -> put result under single "value" section
        result is string -> find first occurrence of single-valued tag (Choices, TextArea, etc.) and put string under corresponding single field (e.g. "choices": ["my_label"])  # noqa
        """
        if isinstance(result, list):
            # full representation of result
            for item in result:
                if not isinstance(item, dict):
                    raise ValidationError('Each item in prediction result should be dict')
            # TODO: check consistency with project.label_config
            return result

        elif isinstance(result, dict):
            # "value" from result
            # TODO: validate value fields according to project.label_config
            for tag, tag_info in project.get_parsed_config().items():
                tag_type = tag_info['type'].lower()
                if tag_type in result:
                    return [
                        {
                            'from_name': tag,
                            'to_name': ','.join(tag_info['to_name']),
                            'type': tag_type,
                            'value': result,
                        }
                    ]

        elif isinstance(result, (str, numbers.Integral)):
            # If result is of integral type, it could be a representation of data from single-valued control tags (e.g. Choices, Rating, etc.)
            for tag, tag_info in project.get_parsed_config().items():
                tag_type = tag_info['type'].lower()
                if tag_type in SINGLE_VALUED_TAGS and isinstance(result, SINGLE_VALUED_TAGS[tag_type]):
                    return [
                        {
                            'from_name': tag,
                            'to_name': ','.join(tag_info['to_name']),
                            'type': tag_type,
                            'value': {tag_type: [result]},
                        }
                    ]
        else:
            raise ValidationError(f'Incorrect format {type(result)} for prediction result {result}')

    def update_task(self):
        update_fields = ['updated_at']

        # updated_by
        request = get_current_request()
        if request:
            self.task.updated_by = request.user
            update_fields.append('updated_by')

        self.task.save(update_fields=update_fields, skip_fsm=True)

    def save(self, *args, update_fields=None, **kwargs):
        if self.project_id is None and self.task_id:
            logger.warning('project_id is not set for prediction, project_id being set in save method')
            self.project_id = Task.objects.only('project_id').get(pk=self.task_id).project_id
            if update_fields is not None:
                update_fields = {'project_id'}.union(update_fields)

        # "result" data can come in different forms - normalize them to JSON
        self.result = self.prepare_prediction_result(self.result, self.project)

        if update_fields is not None:
            update_fields = {'result'}.union(update_fields)
        # set updated_at field of task to now()
        self.update_task()
        return super(Prediction, self).save(*args, update_fields=update_fields, **kwargs)

    def delete(self, *args, **kwargs):
        result = super().delete(*args, **kwargs)
        # set updated_at field of task to now()
        self.update_task()
        return result

    @classmethod
    def create_no_commit(
        cls, project, label_interface, task_id, data, model_version, model_run
    ) -> Optional['Prediction']:
        """
        Creates a Prediction object from the given result data, without committing it to the database.
        or returns None if it fails.

        Args:
            project: The Project object to associate with the Prediction object.
            label_interface: The LabelInterface object to use to create regions from the result data.
            data: The data to create the Prediction object with.
                    Must contain keys that match control tags in labeling configuration
                    Example:
                        ```
                        {
                            'my_choices': 'positive',
                            'my_textarea': 'generated document summary'
                        }
                        ```
            task_id: The ID of the Task object to associate with the Prediction object.
            model_version: The model version that produced the prediction.
            model_run: The model run that created the prediction.
        """
        try:
            # given the data receive, create annotation regions in LS format
            # e.g. {"sentiment": "positive"} -> {"value": {"choices": ["positive"]}, "from_name": "", "to_name": "", ..}
            pred = PredictionValue(result=label_interface.create_regions(data)).model_dump()

            prediction = Prediction(
                project=project,
                task_id=int(task_id),
                model_version=model_version,
                model_run=model_run,
                score=1.0,  # Setting to 1.0 for now as we don't get back a score
                result=pred['result'],
            )
            return prediction
        except Exception as exc:
            # TODO: handle exceptions better
            logger.error(
                f'Error creating prediction for task {task_id} with {data}: {exc}. Traceback: {traceback.format_exc()}'
            )

    class Meta:
        db_table = 'prediction'


class FailedPrediction(models.Model):
    """
    Class for storing failed prediction(s) for a task
    """

    message = models.TextField(
        _('message'),
        default=None,
        blank=True,
        null=True,
        help_text='The message explaining why generating this prediction failed',
    )
    error_type = models.CharField(
        _('error_type'),
        max_length=512,
        default=None,
        null=True,
        help_text='The type of error that caused prediction to fail',
    )
    ml_backend_model = models.ForeignKey(
        'ml.MLBackend',
        on_delete=models.SET_NULL,
        related_name='failed_predictions',
        null=True,
        help_text='An ML Backend instance that created the failed prediction.',
    )
    model_version = models.TextField(
        _('model version'),
        default=None,
        blank=True,
        null=True,
        help_text='A string value that for model version that produced the failed prediction. Used in both live models and when uploading offline predictions.',
    )
    model_run = models.ForeignKey(
        'ml_models.ModelRun',
        on_delete=models.CASCADE,
        related_name='failed_predictions',
        null=True,
        help_text='A run of a ModelVersion that created the failed prediction.',
    )
    project = models.ForeignKey(
        'projects.Project', on_delete=models.CASCADE, related_name='failed_predictions', null=True
    )
    task = models.ForeignKey('tasks.Task', on_delete=models.CASCADE, related_name='failed_predictions')
    created_at = models.DateTimeField(_('created at'), auto_now_add=True)


class PredictionMeta(models.Model):
    prediction = models.OneToOneField(
        'Prediction',
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name='meta',
        help_text=_('Reference to the associated prediction'),
    )
    failed_prediction = models.OneToOneField(
        'FailedPrediction',
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name='meta',
        help_text=_('Reference to the associated failed prediction'),
    )
    inference_time = models.FloatField(
        _('inference time'), null=True, blank=True, help_text=_('Time taken for inference in seconds')
    )
    prompt_tokens_count = models.IntegerField(
        _('prompt tokens count'), null=True, blank=True, help_text=_('Number of tokens in the prompt')
    )
    completion_tokens_count = models.IntegerField(
        _('completion tokens count'), null=True, blank=True, help_text=_('Number of tokens in the completion')
    )
    total_tokens_count = models.IntegerField(
        _('total tokens count'), null=True, blank=True, help_text=_('Total number of tokens')
    )
    prompt_cost = models.FloatField(_('prompt cost'), null=True, blank=True, help_text=_('Cost of the prompt'))
    completion_cost = models.FloatField(
        _('completion cost'), null=True, blank=True, help_text=_('Cost of the completion')
    )
    total_cost = models.FloatField(_('total cost'), null=True, blank=True, help_text=_('Total cost'))
    extra = models.JSONField(_('extra'), null=True, blank=True, help_text=_('Additional metadata in JSON format'))

    @classmethod
    def create_no_commit(cls, data, prediction: Union[Prediction, FailedPrediction]) -> Optional['PredictionMeta']:
        """
        Creates a PredictionMeta object from the given result data, without committing it to the database.
        or returns None if it fails.

        Args:
            data: The data to create the PredictionMeta object with.
                  Example:
                    {
                        'total_cost_usd': 0.1,
                        'prompt_cost_usd': 0.05,
                        'completion_cost_usd': 0.05,
                        'prompt_tokens': 10,
                        'completion_tokens': 10,
                        'inference_time': 0.1
                    }
            prediction: The Prediction or FailedPrediction object to associate with the PredictionMeta object.

        Returns:
            The PredictionMeta object created from the given data, or None if it fails.
        """
        try:
            prediction_meta = PredictionMeta(
                total_cost=data.get('total_cost_usd'),
                prompt_cost=data.get('prompt_cost_usd'),
                completion_cost=data.get('completion_cost_usd'),
                prompt_tokens_count=data.get('prompt_tokens'),
                completion_tokens_count=data.get('completion_tokens'),
                total_tokens_count=data.get('prompt_tokens', 0) + data.get('completion_tokens', 0),
                inference_time=data.get('inference_time'),
                extra={
                    'message_counts': data.get('message_counts', {}),
                },
            )
            if isinstance(prediction, Prediction):
                prediction_meta.prediction = prediction
            else:
                prediction_meta.failed_prediction = prediction
            logger.debug(f'Created prediction meta: {prediction_meta}')
            return prediction_meta
        except Exception as exc:
            logger.error(f'Error creating prediction meta with {data}: {exc}. Traceback: {traceback.format_exc()}')

    class Meta:
        db_table = 'prediction_meta'
        verbose_name = _('Prediction Meta')
        verbose_name_plural = _('Prediction Metas')
        constraints = [
            CheckConstraint(
                # either prediction or failed_prediction should be not null
                check=(
                    (Q(prediction__isnull=False) & Q(failed_prediction__isnull=True))
                    | (Q(prediction__isnull=True) & Q(failed_prediction__isnull=False))
                ),
                name='prediction_or_failed_prediction_not_null',
            )
        ]

# tasks/models.py에 병합 시 이 import는 파일 상단에 이미 존재:
# from tasks.models import Task  ← 병합 시 제거하고 Task 직접 참조
DJI_RTK_FIXED = 50

class TaskGeoreferencing(models.Model):
    """Task 1:1 georeferencing 데이터.
 
    소스: DJI XMP + EXIF (dji_metadata_extractor) → 비동기 job이 채움.
    파생: footprint_* (핀홀 광선-지면 교차), *_corrected (지도 보정 반영).
    원본 좌표는 불변, 보정은 GeoAlignmentCorrection 저장 시 재계산해 물질화.
    """
 
    class Method(models.TextChoices):
        RTK = 'RTK', 'RTK'
        PPK = 'PPK', 'PPK'
        GCP = 'GCP', 'GCP-based'
        NONE = 'NONE', 'None'
 
    class Status(models.TextChoices):
        """EXIF/XMP 비동기 추출 파이프라인 상태."""
        PENDING = 'pending', 'Pending extraction'
        COMPLETED = 'completed', 'Completed'
        NO_GPS = 'no_gps', 'No GPS metadata'
        UNPAIRED = 'unpaired', 'RGB/IR pair not found'
        FAILED = 'failed', 'Extraction failed'
 
    class GroundAltSource(models.TextChoices):
        LRF = 'lrf', 'Laser Range Finder (실측)'
        RELATIVE = 'relative', 'AbsoluteAlt - RelativeAlt'
        DEM = 'dem', 'DEM lookup'
        NONE = 'none', 'Unknown'
 
    # ------------------------------------------------------------------
    # 관계 (단일 정의 — 기존 중복 버그 수정)
    # ------------------------------------------------------------------
    task = models.OneToOneField(
        'tasks.Task',
        on_delete=models.CASCADE,
        related_name='georeferencing',
        help_text='1:1 대상 task',
    )
 
    # ------------------------------------------------------------------
    # 위치/자세 (기존 필드명 유지 — serializer 호환)
    # ------------------------------------------------------------------
    latitude = models.FloatField(null=True, blank=True, help_text='WGS84 [deg]')
    longitude = models.FloatField(null=True, blank=True, help_text='WGS84 [deg]')
    altitude = models.FloatField(
        null=True, blank=True, help_text='절대고도(타원체고) [m]')
    relative_height = models.FloatField(
        null=True, blank=True, help_text='이륙점 기준 상대고도 [m]')
 
    yaw = models.FloatField(null=True, blank=True, help_text='짐벌 yaw [deg], 북 기준 시계+')
    pitch = models.FloatField(null=True, blank=True, help_text='짐벌 pitch [deg], nadir=-90')
    roll = models.FloatField(null=True, blank=True, help_text='짐벌 roll [deg]')
    flight_yaw = models.FloatField(
        null=True, blank=True, help_text='기체 yaw [deg] — 짐벌과 별도')
 
    # ------------------------------------------------------------------
    # 지면고도 (ray-plane intersection의 ground_z)
    # ------------------------------------------------------------------
    ground_alt = models.FloatField(
        null=True, blank=True, help_text='지면 절대고도 [m]')
    ground_alt_source = models.CharField(
        max_length=8, choices=GroundAltSource.choices,
        default=GroundAltSource.NONE,
        help_text='LRF 실측이 relative 역산보다 정확 — 우선 사용')
 
    # ------------------------------------------------------------------
    # RTK 품질
    # ------------------------------------------------------------------
    rtk_flag = models.IntegerField(
        null=True, blank=True,
        help_text='DJI RtkFlag: 0=GPS only, 16=Float, 50=Fixed(신뢰 가능)')
    rtk_std_lat = models.FloatField(null=True, blank=True, help_text='[m]')
    rtk_std_lon = models.FloatField(null=True, blank=True, help_text='[m]')
    rtk_std_hgt = models.FloatField(null=True, blank=True, help_text='[m]')
 
    # ------------------------------------------------------------------
    # 카메라 / 촬영
    # ------------------------------------------------------------------
    captured_at = models.DateTimeField(
        null=True, blank=True, db_index=True,
        help_text='촬영시각 — RGB/IR 페어링·시계열 추적의 기준')
    camera_model = models.CharField(max_length=64, blank=True, default='')
    modality = models.CharField(
        max_length=8, blank=True, default='',
        help_text='wide | zoom | ir (파일명 접미사 _W/_Z/_T 기반)')
    focal_length_mm = models.FloatField(null=True, blank=True)
    focal_length_35mm = models.IntegerField(
        null=True, blank=True, help_text='35mm 환산 — FOV/footprint 계산 입력')
    image_width = models.IntegerField(null=True, blank=True)
    image_height = models.IntegerField(null=True, blank=True)
    image_sha1 = models.CharField(
        max_length=40, blank=True, default='', db_index=True,
        help_text='파일 SHA-1 (metadata.py 컨벤션) — SQLite 인덱스와 조인 키')
 
    # ------------------------------------------------------------------
    # georeferencing 결과 (기존 필드 유지)
    # ------------------------------------------------------------------
    ground_sample_distance = models.FloatField(
        null=True, blank=True, help_text='GSD [m/px]')
    method = models.CharField(
        max_length=8, choices=Method.choices, default=Method.NONE)
    accuracy_m = models.FloatField(
        null=True, blank=True, help_text='추정 수평 정확도 [m]')
    offset_correction = models.JSONField(
        null=True, blank=True,
        help_text='RGB-IR co-registration 보정값 (고도/자세 의존 affine)')
    transform_matrix = models.JSONField(
        null=True, blank=True, help_text='픽셀→지리 변환 행렬')
    raw_metadata = models.JSONField(
        null=True, blank=True,
        help_text='ImageMetadata.to_dict() 전체 — 재계산/디버깅용 원본 보존')
 
    # ------------------------------------------------------------------
    # footprint (파생 — 원본, 보정본 분리 물질화)
    # ------------------------------------------------------------------
    footprint_rgb = gis.PolygonField(
        srid=4326, null=True, blank=True,
        help_text='RGB 지면 커버리지 (핀홀 광선-지면 교차, 원본 좌표)')
    footprint_ir = gis.PolygonField(
        srid=4326, null=True, blank=True,
        help_text='IR 커버리지 — 광학축 오프셋/FOV가 달라 RGB와 별도')
    footprint_rgb_corrected = gis.PolygonField(
        srid=4326, null=True, blank=True,
        help_text='GeoAlignmentCorrection 반영본 — 공간 검색은 이 컬럼 사용')
    footprint_ir_corrected = gis.PolygonField(
        srid=4326, null=True, blank=True)
    footprint_low_confidence = models.BooleanField(
        default=False,
        help_text='near-horizon clamp 등으로 footprint 신뢰 낮음')
 
    # ------------------------------------------------------------------
    # 파이프라인 관리
    # ------------------------------------------------------------------
    status = models.CharField(
        max_length=16, choices=Status.choices, default=Status.PENDING,
        db_index=True, help_text='비동기 EXIF/XMP 추출 상태')
    extractor_version = models.CharField(
        max_length=16, blank=True, default='',
        help_text='추출 로직 버전 — 업그레이드 후 선별 재계산 대상 조회용')
    error_message = models.TextField(
        blank=True, default='', help_text='status=failed일 때 원인')
 
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
 
    class Meta:
        # 기존 데이터 이전 없이 그대로 쓰려면 'georeferencing_data'로 유지
        db_table = 'task_georeferencing'
        indexes = [
            GistIndex(fields=['footprint_rgb_corrected'],
                      name='taskgeo_fp_rgb_corr_gist'),
            GistIndex(fields=['footprint_ir_corrected'],
                      name='taskgeo_fp_ir_corr_gist'),
            models.Index(fields=['status', 'extractor_version'],
                         name='taskgeo_status_ver'),
            models.Index(fields=['modality', 'captured_at'],
                         name='taskgeo_modality_time'),
        ]
 
    def __str__(self):
        return (f'TaskGeoreferencing(task={self.task_id}, '
                f'{self.status}, ({self.latitude}, {self.longitude}), '
                f'rtk={self.rtk_flag})')
 
    # ------------------------------------------------------------------
    # 헬퍼
    # ------------------------------------------------------------------
 
    @property
    def is_rtk_fixed(self) -> bool:
        return self.rtk_flag == DJI_RTK_FIXED
 
    @property
    def is_ready(self) -> bool:
        """georeferencing 계산 투입 가능 여부."""
        return (self.status == self.Status.COMPLETED
                and self.latitude is not None
                and self.longitude is not None
                and self.yaw is not None)
 
    @property
    def camera_height_m(self):
        """지면 기준 카메라 높이 [m] — footprint/ray-cast 입력."""
        if self.altitude is not None and self.ground_alt is not None:
            return self.altitude - self.ground_alt
        return self.relative_height
 
    def populate_from_metadata(self, geo_dict: dict, extractor_version: str = '1'):
        """dji_metadata_extractor.to_georeferencing_dict() 결과를 반영한다.
 
        사용 (비동기 job 내부):
 
            meta = extract(image_path)
            geo = TaskGeoreferencing.objects.get_or_create(task=task)[0]
            geo.populate_from_metadata(to_georeferencing_dict(meta))
            geo.raw_metadata = meta.to_dict()   # ndarray 필드는 호출부에서 제거
            geo.save()
        """
        self.latitude = geo_dict.get('rtk_lat')
        self.longitude = geo_dict.get('rtk_lon')
        self.altitude = geo_dict.get('rtk_alt')
        self.relative_height = geo_dict.get('relative_height')
        self.ground_alt = geo_dict.get('ground_alt')
        self.ground_alt_source = geo_dict.get(
            'ground_alt_source', self.GroundAltSource.NONE)
        self.yaw = geo_dict.get('yaw')
        self.pitch = geo_dict.get('pitch')
        self.roll = geo_dict.get('roll')
        self.flight_yaw = geo_dict.get('flight_yaw')
        self.rtk_flag = geo_dict.get('rtk_flag')
        self.rtk_std_lat = geo_dict.get('rtk_std_lat')
        self.rtk_std_lon = geo_dict.get('rtk_std_lon')
        self.rtk_std_hgt = geo_dict.get('rtk_std_hgt')
        self.captured_at = geo_dict.get('captured_at')
        self.camera_model = geo_dict.get('camera_model') or ''
        self.modality = geo_dict.get('modality') or ''
        self.focal_length_mm = geo_dict.get('focal_length_mm')
        self.focal_length_35mm = geo_dict.get('focal_length_35mm')
        self.image_width = geo_dict.get('image_width')
        self.image_height = geo_dict.get('image_height')
        self.image_sha1 = geo_dict.get('image_sha1') or ''
        self.method = (self.Method.RTK if self.is_rtk_fixed
                       else self.Method.NONE)
        self.status = (self.Status.COMPLETED
                       if self.latitude and self.longitude
                       else self.Status.NO_GPS)
        self.extractor_version = extractor_version
        self.error_message = ''
 
    def mark_failed(self, message: str):
        self.status = self.Status.FAILED
        self.error_message = str(message)[:2000]
 
    def footprint_for(self, modality: str, corrected: bool = True):
        """검색/표시에 쓸 footprint 선택. 보정본 우선, 없으면 원본."""
        if modality == 'ir':
            return ((self.footprint_ir_corrected if corrected else None)
                    or self.footprint_ir)
        return ((self.footprint_rgb_corrected if corrected else None)
                or self.footprint_rgb)


class TaskImageGeoreferencing(models.Model):
    """XMP+EXIF 추출 사이드카 인덱스의 이미지 1장짜리 레코드.
 
    GpsImageIndexPG._SCHEMA가 만드는 원시 테이블의 ORM 미러.
    id는 파일 SHA-1(metadata.py 컨벤션)이라 같은 task라도 RGB/IR이
    서로 다른 row를 가진다 (TaskGeoreferencing과 달리 1:N).
    """
 
    class Modality(models.TextChoices):
        WIDE = 'wide', 'Wide (RGB)'
        ZOOM = 'zoom', 'Zoom (RGB)'
        IR = 'ir', 'Infrared'
        SCREEN = 'screen', 'Screenshot'
 
    class GroundAltSource(models.TextChoices):
        LRF = 'lrf', 'Laser Range Finder (실측)'
        RELATIVE = 'relative', 'AbsoluteAlt - RelativeAlt'
        DEM = 'dem', 'DEM lookup'
        NONE = 'none', 'Unknown'
 
    # ------------------------------------------------------------------
    # 식별자
    # ------------------------------------------------------------------
    id = models.CharField(
        primary_key=True, max_length=40,
        help_text='파일 SHA-1 (metadata.py _compute_id 컨벤션)')
 
    # DB 컬럼은 task_id/project_id 그대로 두고, ORM에서는 FK로 접근.
    # 원본 인덱스는 Label Studio task 존재 여부와 무관하게 채워질 수 있어
    # null=True — CASCADE는 Task 삭제 시 사이드카도 함께 정리.
    task = models.ForeignKey(
        'tasks.Task', on_delete=models.CASCADE,
        db_column='task_id', null=True, blank=True,
        related_name='image_georeferencing_records',
        help_text='이 이미지가 속한 Task (1 task = RGB/IR 등 여러 row 가능)')
    project = models.ForeignKey(
        'projects.Project', on_delete=models.CASCADE,
        db_column='project_id', null=True, blank=True,
        related_name='image_georeferencing_records')
 
    path = models.TextField(help_text='인덱싱 시점의 디스크 경로')
    filename = models.CharField(max_length=255)
    modality = models.CharField(
        max_length=8, choices=Modality.choices, blank=True, default='')
 
    capture_time = models.BigIntegerField(
        null=True, blank=True,
        help_text='epoch seconds — TaskGeoreferencing.captured_at은 '
                  'DateTimeField이므로 표시 시 변환 필요')

    def populate_from_metadata(self, geo_dict: dict, extractor_version: str = '1'):
        """dji_metadata_extractor.to_georeferencing_dict() 결과를 반영한다.
 
        사용 (비동기 job 내부):
 
            meta = extract(image_path)
            geo = TaskImageGeoreferencing.objects.get_or_create(task=task, project=project)[0]
            geo.populate_from_metadata(meta)
            geo.save()
        """
        self.id = geo_dict.get('id')
        self.path = geo_dict.get('path')
        self.filename = geo_dict.get('filename')
        self.modality = geo_dict.get('modality')
        self.capture_time = geo_dict.get('capture_time')
        self.lat = geo_dict.get('lat')
        self.lng = geo_dict.get('lng')
        self.altitude = geo_dict.get('altitude')
        self.relative_height = geo_dict.get('relative_height')
        self.ground_alt = geo_dict.get('ground_alt')
        self.ground_alt_src = geo_dict.get(
            'ground_alt_src', self.GroundAltSource.NONE)
        self.gimbal_yaw = geo_dict.get('gimbal_yaw')
        self.gimbal_pitch = geo_dict.get('gimbal_pitch')
        self.gimbal_roll = geo_dict.get('gimbal_roll')
        self.rtk_flag = geo_dict.get('rtk_flag')
        self.width = geo_dict.get('width')
        self.height = geo_dict.get('height')
        self.focal_35mm = geo_dict.get("focal_35mm")
        self.low_confidence = geo_dict.get("low_confidence")
        self.footprint_geojson = geo_dict.get('footprint_geojson')
        self.meta_json = geo_dict.get('meta_json')

    # ------------------------------------------------------------------
    # 위치/자세 (원본 XMP 값 — 보정 미반영)
    # ------------------------------------------------------------------
    lat = models.FloatField(null=True, blank=True)
    lng = models.FloatField(null=True, blank=True)
    altitude = models.FloatField(null=True, blank=True, help_text='절대고도 [m]')
    relative_height = models.FloatField(null=True, blank=True, help_text='[m]')
 
    ground_alt = models.FloatField(null=True, blank=True)
    ground_alt_src = models.CharField(
        max_length=8, choices=GroundAltSource.choices,
        blank=True, default=GroundAltSource.NONE)
 
    gimbal_yaw = models.FloatField(null=True, blank=True)
    gimbal_pitch = models.FloatField(null=True, blank=True)
    gimbal_roll = models.FloatField(null=True, blank=True)
 
    rtk_flag = models.IntegerField(
        null=True, blank=True,
        help_text='DJI RtkFlag: 0=GPS only, 16=Float, 50=Fixed')
 
    # ------------------------------------------------------------------
    # 카메라
    # ------------------------------------------------------------------
    width = models.IntegerField(null=True, blank=True)
    height = models.IntegerField(null=True, blank=True)
    focal_35mm = models.IntegerField(null=True, blank=True)
 
    # ------------------------------------------------------------------
    # footprint (단일 — TaskGeoreferencing과 달리 보정본 분리 없음.
    # 이 사이드카는 원본 XMP 좌표 기준 "원본 이미지 찾기" 용도이므로
    # GeoAlignmentCorrection 반영은 TaskGeoreferencing.footprint_*_corrected가
    # 맡는다.)
    # ------------------------------------------------------------------
    low_confidence = models.BooleanField(
        default=False,
        help_text='near-horizon clamp 등으로 footprint 신뢰 낮음')
    footprint = gis.PolygonField(
        srid=4326,
        help_text='핀홀 광선-지면 교차 (원본 좌표, NOT NULL)')
 
    meta_json = models.JSONField(
        null=True, blank=True,
        help_text='ImageMetadata.to_dict() 전체 — 재계산/디버깅용 원본 보존')
 
    indexed_at = models.DateTimeField(
        auto_now_add=True,
        help_text='raw SQL의 DEFAULT now()에 대응')
 
    class Meta:
        # 이 테이블의 DDL 소유자는 gps_image_index_pg.py의 _SCHEMA 스크립트다.
        # Django 마이그레이션이 CREATE/ALTER/인덱스 관리를 하지 않도록 고정.
        managed = False
        #db_table = 'public"."task_image_georeferencing'
        db_table = 'task_image_georeferencing'
        indexes = [
            # 아래 인덱스는 문서화 목적 — managed=False이므로 실제 생성/변경은
            # 이 클래스가 아니라 _SCHEMA 스크립트가 수행한다.
            GistIndex(fields=['footprint'], name='idx_gpsimg_footprint'),
            models.Index(fields=['task'], name='idx_gpsimg_task'),
            models.Index(fields=['project'], name='idx_gpsimg_project'),
            models.Index(fields=['capture_time'], name='idx_gpsimg_capture'),
            models.Index(fields=['modality'], name='idx_gpsimg_modality'),
        ]
 
    def __str__(self):
        return (f'TaskImageGeoreferencing(id={self.id[:8]}…, '
                f'task={self.task_id}, {self.modality}, '
                f'({self.lat}, {self.lng}), rtk={self.rtk_flag})')
 
    # ------------------------------------------------------------------
    # 헬퍼 (TaskGeoreferencing과 동일 컨벤션)
    # ------------------------------------------------------------------
 
    @property
    def is_rtk_fixed(self) -> bool:
        return self.rtk_flag == 50
 
    @property
    def camera_height_m(self):
        if self.altitude is not None and self.ground_alt is not None:
            return self.altitude - self.ground_alt
        return self.relative_height
 
    def to_task_georeferencing_defaults(self) -> dict:
        """TaskGeoreferencing.populate_from_metadata()와 동일한 키 매핑.
 
        이 사이드카 레코드로 TaskGeoreferencing을 채우거나 검증할 때 사용.
        """
        return {
            'rtk_lat': self.lat,
            'rtk_lon': self.lng,
            'rtk_alt': self.altitude,
            'relative_height': self.relative_height,
            'ground_alt': self.ground_alt,
            'ground_alt_source': self.ground_alt_src,
            'yaw': self.gimbal_yaw,
            'pitch': self.gimbal_pitch,
            'roll': self.gimbal_roll,
            'rtk_flag': self.rtk_flag,
            'camera_model': '',
            'modality': self.modality,
            'focal_length_35mm': self.focal_35mm,
            'image_width': self.width,
            'image_height': self.height,
            'image_sha1': self.id,
        }


@receiver(post_delete, sender=Task)
def update_all_task_states_after_deleting_task(sender, instance, **kwargs):
    """after deleting_task
    use update_tasks_states for all project
    but call only tasks_number_changed section
    """
    try:
        instance.project.update_tasks_states(
            maximum_annotations_changed=False,
            overlap_cohort_percentage_changed=False,
            tasks_number_changed=True,
        )
    except Exception as exc:
        logger.error('Error in update_all_task_states_after_deleting_task: ' + str(exc))


# =========== PROJECT SUMMARY UPDATES ===========


@receiver(pre_delete, sender=Task)
def remove_data_columns(sender, instance, **kwargs):
    """Reduce data column counters after removing task"""
    instance.decrease_project_summary_counters()


def _task_data_is_not_updated(update_fields):
    if update_fields and list(update_fields) == ['is_labeled']:
        return True


@receiver(pre_save, sender=Task)
def delete_project_summary_data_columns_before_updating_task(sender, instance, update_fields, **kwargs):
    """Before updating task fields - ensure previous info removed from project.summary"""
    if _task_data_is_not_updated(update_fields):
        # we don't need to update counters when other than task.data fields are updated
        return
    try:
        old_task = sender.objects.get(id=instance.id)
    except Task.DoesNotExist:
        # task just created - do nothing
        return
    old_task.decrease_project_summary_counters()


@receiver(post_save, sender=Task)
def update_project_summary_data_columns(sender, instance, created, update_fields, **kwargs):
    """Update task counters in project summary in case when new task has been created"""
    if _task_data_is_not_updated(update_fields):
        # we don't need to update counters when other than task.data fields are updated
        return
    instance.increase_project_summary_counters()


@receiver(pre_save, sender=Annotation)
def delete_project_summary_annotations_before_updating_annotation(sender, instance, **kwargs):
    """Before updating annotation fields - ensure previous info removed from project.summary"""
    try:
        old_annotation = sender.objects.get(id=instance.id)
    except Annotation.DoesNotExist:
        # annotation just created - do nothing
        return
    old_annotation.decrease_project_summary_counters()

    # update task counters if annotation changes it's was_cancelled status
    task = instance.task
    if old_annotation.was_cancelled != instance.was_cancelled:
        if instance.was_cancelled:
            task.cancelled_annotations = task.cancelled_annotations + 1
            task.total_annotations = task.total_annotations - 1
        else:
            task.cancelled_annotations = task.cancelled_annotations - 1
            task.total_annotations = task.total_annotations + 1
        task.update_is_labeled()

        Task.objects.filter(id=instance.task.id).update(
            is_labeled=task.is_labeled,
            total_annotations=task.total_annotations,
            cancelled_annotations=task.cancelled_annotations,
        )


@receiver(post_save, sender=Annotation)
def update_project_summary_annotations_and_is_labeled(sender, instance, created, **kwargs):
    """Update annotation counters in project summary"""
    instance.increase_project_summary_counters()

    # If annotation is changed, update task.is_labeled state
    logger.debug(f'Update task stats for task={instance.task}')
    if instance.was_cancelled:
        instance.task.cancelled_annotations = instance.task.annotations.all().filter(was_cancelled=True).count()
    else:
        instance.task.total_annotations = instance.task.annotations.all().filter(was_cancelled=False).count()
    instance.task.update_is_labeled()
    instance.task.save(update_fields=['is_labeled', 'total_annotations', 'cancelled_annotations'], skip_fsm=True)
    logger.debug(f'Updated total_annotations and cancelled_annotations for {instance.task.id}.')


@receiver(pre_delete, sender=Prediction)
def remove_predictions_from_project(sender, instance, **kwargs):
    """Remove predictions counters"""
    instance.task.total_predictions = instance.task.predictions.all().count() - 1
    instance.task.save(update_fields=['total_predictions'])
    logger.debug(f'Updated total_predictions for {instance.task.id}.')


@receiver(post_save, sender=Prediction)
def save_predictions_to_project(sender, instance, **kwargs):
    """Add predictions counters"""
    instance.task.total_predictions = instance.task.predictions.all().count()
    instance.task.save(update_fields=['total_predictions'])
    logger.debug(f'Updated total_predictions for {instance.task.id}.')


# =========== END OF PROJECT SUMMARY UPDATES ===========


@receiver(post_save, sender=Annotation)
def delete_draft(sender, instance, **kwargs):
    task = instance.task
    query_args = {'task': task, 'annotation': instance}
    drafts = AnnotationDraft.objects.filter(**query_args)
    num_drafts = 0
    for draft in drafts:
        # Delete each draft individually because deleting
        # the whole queryset won't update `created_labels_drafts`
        try:
            draft.delete()
            num_drafts += 1
        except AnnotationDraft.DoesNotExist:
            continue
    logger.debug(f'{num_drafts} drafts removed from task {task} after saving annotation {instance}')


@receiver(post_save, sender=Annotation)
def update_ml_backend(sender, instance, **kwargs):
    if instance.ground_truth:
        return

    project = instance.project

    if hasattr(project, 'ml_backends') and project.min_annotations_to_start_training:
        annotation_count = Annotation.objects.filter(project=project).count()

        # start training every N annotation
        if annotation_count % project.min_annotations_to_start_training == 0:
            for ml_backend in project.ml_backends.all():
                ml_backend.train()


def update_task_stats(task, stats=('is_labeled',), save=True):
    """Update single task statistics:
        accuracy
        is_labeled
    :param task: Task to update
    :param stats: to update separate stats
    :param save: to skip saving in some cases
    :return:
    """
    logger.debug(f'Update stats {stats} for task {task}')
    if 'is_labeled' in stats:
        task.update_is_labeled(save=save)
    if save:
        task.save()


def deprecated_bulk_update_stats_project_tasks(tasks, project=None):
    """bulk Task update accuracy
       ex: after change settings
       apply several update queries size of batch
       on updated Task objects
       in single transaction as execute sql
    :param tasks:
    :return:
    """
    # recalc accuracy
    if not tasks:
        # break if tasks is empty
        return
    # get project if it's not in params
    if project is None:
        project = tasks[0].project

    with transaction.atomic():
        use_overlap = project._can_use_overlap()
        # update filters if we can use overlap
        if use_overlap:
            # following definition of `completed_annotations` above, count cancelled annotations
            # as completed if project is in IGNORE_SKIPPED mode
            completed_annotations_f_expr = F('total_annotations')
            if project.skip_queue == project.SkipQueue.IGNORE_SKIPPED:
                completed_annotations_f_expr += F('cancelled_annotations')
            finished_q = Q(GreaterThanOrEqual(completed_annotations_f_expr, F('overlap')))
            finished_tasks = tasks.filter(finished_q)
            finished_tasks_ids = finished_tasks.values_list('id', flat=True)
            tasks.update(is_labeled=Q(id__in=finished_tasks_ids))

        else:
            # update objects without saving if we can't use overlap
            for task in tasks:
                update_task_stats(task, save=False)
            try:
                # start update query batches
                bulk_update(tasks, update_fields=['is_labeled'], batch_size=settings.BATCH_SIZE)
            except OperationalError:
                logger.error('Operational error while updating tasks: {exc}', exc_info=True)
                # try to update query batches one more time
                start_job_async_or_sync(
                    bulk_update,
                    tasks,
                    in_seconds=settings.BATCH_JOB_RETRY_TIMEOUT,
                    update_fields=['is_labeled'],
                    batch_size=settings.BATCH_SIZE,
                )


def bulk_update_stats_project_tasks(tasks, project=None):
    # Avoid circular import
    from projects.functions.utils import get_unique_ids_list

    bulk_update_is_labeled = load_func(settings.BULK_UPDATE_IS_LABELED)

    if flag_set('fflag_fix_back_plt_802_update_is_labeled_20062025_short', user='auto'):
        task_ids = get_unique_ids_list(tasks)

        if not task_ids:
            return

        if project is None:
            first_task = Task.objects.get(id=task_ids[0])
            project = first_task.project

        # Set user context so FSM checks work when this runs in an async worker
        if project.created_by_id:
            CurrentContext.set_user(project.created_by)
        bulk_update_is_labeled(task_ids, project)
    else:
        return deprecated_bulk_update_stats_project_tasks(tasks, project)


Q_finished_annotations = Q(was_cancelled=False) & Q(result__isnull=False)
Q_task_finished_annotations = Q(annotations__was_cancelled=False) & Q(annotations__result__isnull=False)
