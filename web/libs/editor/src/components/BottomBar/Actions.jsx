import { useState, useCallback } from "react";
import { IconInfoOutline, IconPredictions, IconSettings } from "@humansignal/icons";
import { Button } from "@humansignal/ui";
import { isStarterCloudPlan } from "@humansignal/core";
import { cn } from "../../utils/bem";
import { FF_BULK_ANNOTATION, isFF } from "../../utils/feature-flags";
import { AutoAcceptToggle } from "../AnnotationTab/AutoAcceptToggle";
import { DynamicPreannotationsToggle } from "../AnnotationTab/DynamicPreannotationsToggle";
import { GroundTruth } from "../CurrentEntity/GroundTruth";
import { EditingHistory } from "./HistoryActions";
import { RepredictButton } from "./RepredictButton";  // ✅ 분리된 컴포넌트 import
import { ExcludeExportButton } from "./ExcludeExportButton";
import { GeoreferencingButton } from "./GeoreferencingButton";
import "./Actions.prefix.css";

export const Actions = ({ store }) => {
  const annotationStore = store.annotationStore;
  const entity = annotationStore.selected;
  const isPrediction = entity?.type === "prediction";
  const isViewAll = annotationStore.viewingAll === true;
  const isBulkMode = isFF(FF_BULK_ANNOTATION) && !isStarterCloudPlan() && store.hasInterface("annotation:bulk");

  // ✅ 재예측 상태 및 핸들러
  const [isRepredicting, setIsRepredicting] = useState(false);

  // ── 실제 재예측 실행 ──────────────────────────────────────────
  const executeRepredict = useCallback(
    async (taskId, projectId) => {
      setIsRepredicting(true);
      try {
        const response = await fetch(`/api/dm/actions?id=retrieve_tasks_predictions&project=${projectId}`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-CSRFToken": document.cookie.match(/csrftoken=([^;]+)/)?.[1] ?? "",
          },
          body: JSON.stringify({
            selectedItems: { all: false, included: [taskId] },
          }),
        });

        if (!response.ok) {
          const text = await response.text();
          throw new Error(`HTTP ${response.status}: ${text}`);
        }

        if (store.SDK?.invoke) {
          store.SDK.invoke("taskReload", { taskId });
        }
        window.location.reload();
      } catch (e) {
        console.error("Repredict error:", e);
        alert(`재예측 실패: ${e.message}`);
      } finally {
        setIsRepredicting(false);
      }
    },
    [store],
  );

  // ── 확인 Dialog 표시 (ActionsButton.jsx invokeAction 패턴) ──────
  const handleRepredict = useCallback(() => {
    if (isRepredicting) return;

    const taskId = store.task?.id;
    if (!taskId) {
      console.warn("Repredict: no task id");
      return;
    }

    const projectId = window.location.pathname.match(/projects\/(\d+)/)?.[1];
    if (!projectId) {
      console.warn("Repredict: no project id");
      return;
    }

    // ML Backend 모델 버전 표시용 (있으면)
    const modelVersion = store.project?.model_version ?? null;

    // ✅ ActionsButton.jsx의 dialog 호출 패턴 그대로 적용
    Modal.confirm({
      title: "재예측 확인",
      body: <RepredictDialogContent taskId={taskId} modelVersion={modelVersion} />,
      buttonLook: "primary",
      okText: "재예측 실행",
      cancelText: "취소",
      onOk() {
        executeRepredict(taskId, projectId);
      },
      closeOnClickOutside: false,
    });
  }, [store, isRepredicting, executeRepredict]);

  return (
    <div className={cn("bottombar").elem("section").toClassName()}>
      {!isPrediction && !isViewAll && store.hasInterface("edit-history") && <EditingHistory entity={entity} />}

      <div className={cn("action-buttons").toClassName()}>
        {store.description && store.hasInterface("instruction") && (
          <Button
            type="text"
            aria-label="Instructions"
            size="small"
            variant="neutral"
            look="string"
            tooltip="Show instructions"
            onClick={() => store.toggleDescription()}
            className="aspect-square"
            leading={<IconInfoOutline />}
            data-testid="bottombar-instructions-button"
          />
        )}

        {/* ✅ 재예측 버튼 */}
        <RepredictButton store={store} />

        {/* ✅ Georeferencing 버튼 */}
        <GeoreferencingButton store={store} variant="labeled" />

        {/* Export 제외 버튼 */}
        <ExcludeExportButton store={store} />

        <Button
          type="text"
          aria-label="Settings"
          size="small"
          look="string"
          variant="neutral"
          onClick={() => store.toggleSettings()}
          tooltip="Settings"
          className="aspect-square"
          leading={<IconSettings />}
          data-testid="bottombar-settings-button"
        />
      </div>

      {store.hasInterface("ground-truth") && !isBulkMode && <GroundTruth entity={entity} />}

      {!isViewAll && (
        <div className={cn("model-actions").toClassName()}>
          <DynamicPreannotationsToggle />
          <AutoAcceptToggle />
        </div>
      )}
    </div>
  );
};
