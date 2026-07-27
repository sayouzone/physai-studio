// web/libs/editor/src/components/Common/saveDraft.js
//
// georeferencing 검색 전에 현재 annotation의 draft를 서버에 반영한다.
//
// LSF Annotation 모델(Annotation.js) 실제 API 기준으로 작성됨:
//
//   saveDraftImmediately()            → autosave throttle을 flush만 한다.
//                                       Promise를 반환하지 않고, autosave가
//                                       없으면 아무 것도 하지 않는다. 저장 보장 X
//   saveDraftImmediatelyWithResults() → 실제로 saveDraft()를 await 한다. ★사용
//   saveDraft(params)                 → 직렬화 후 store.submitDraft() 호출
//
//   가드: submissionStarted(제출 시작), editable === false(예측/히스토리),
//         isDraftSaving(저장 진행 중이면 {} 반환)
//   상태: draftId 기본값 0, draftSaved(ISO 문자열), isDraftSaving

/** LSF annotation 객체 얻기 */
export function getAnnotation(region, root) {
  return region?.annotation ?? root?.annotationStore?.selected ?? null;
}

/** 조건이 참이 될 때까지 대기 (최대 timeoutMs) */
function waitFor(predicate, timeoutMs = 3000, intervalMs = 50) {
  return new Promise((resolve) => {
    const start = Date.now();
    const tick = () => {
      if (predicate()) return resolve(true);
      if (Date.now() - start >= timeoutMs) return resolve(false);
      setTimeout(tick, intervalMs);
    };
    tick();
  });
}

/**
 * 현재 annotation의 draft를 서버에 저장한다.
 *
 * @param {object} region  LSF region
 * @param {object} root    getRoot(region) 결과
 * @returns {Promise<{saved: boolean, method: string, draftId: number|null, detail?: string}>}
 */
export async function saveDraftBeforeRequest(region, root) {
  const annotation = getAnnotation(region, root);
  if (!annotation) {
    return { saved: false, method: "no-annotation", draftId: null };
  }

  // ── Annotation.saveDraft()의 가드와 동일 조건을 먼저 확인 ──
  // 이 조건들에 걸리면 어떤 메서드를 불러도 조용히 반환하므로,
  // 호출 전에 판별해 사유를 명확히 남긴다.
  if (annotation.submissionStarted) {
    return {
      saved: false,
      method: "submission-started",
      draftId: annotation.draftId || null,
      detail: "제출이 진행 중이라 draft를 저장하지 않습니다.",
    };
  }
  if (annotation.editable === false) {
    return {
      saved: false,
      method: "not-editable",
      draftId: annotation.draftId || null,
      detail:
        "예측(prediction) 또는 히스토리 항목이라 draft 대상이 아닙니다. " +
        "예측을 복사해 annotation으로 만든 뒤 다시 시도하세요.",
    };
  }

  // 진행 중인 저장이 있으면 끝날 때까지 대기.
  // saveDraftImmediatelyWithResults는 isDraftSaving이면 {}만 반환하고 끝난다.
  if (annotation.isDraftSaving) {
    await waitFor(() => !annotation.isDraftSaving, 3000);
  }

  const before = {
    draftId: annotation.draftId ?? 0,
    draftSaved: annotation.draftSaved,
  };

  // ── 1순위: saveDraftImmediatelyWithResults ──
  // 유일하게 실제 저장을 await 하는 메서드.
  if (typeof annotation.saveDraftImmediatelyWithResults === "function") {
    try {
      await annotation.saveDraftImmediatelyWithResults();
      await waitFor(() => !annotation.isDraftSaving, 3000);
      return verify(annotation, before, "saveDraftImmediatelyWithResults");
    } catch (e) {
      console.warn("[draft] saveDraftImmediatelyWithResults 실패", e);
    }
  }

  // ── 2순위: saveDraft 직접 호출 ──
  if (typeof annotation.saveDraft === "function") {
    try {
      await annotation.saveDraft();
      await waitFor(() => !annotation.isDraftSaving, 3000);
      return verify(annotation, before, "saveDraft");
    } catch (e) {
      console.warn("[draft] saveDraft 실패", e);
    }
  }

  // ── 3순위: autosave flush (저장 보장은 없지만 없는 것보단 낫다) ──
  if (typeof annotation.saveDraftImmediately === "function" && annotation.autosave) {
    try {
      annotation.saveDraftImmediately();
      await waitFor(
        () => !annotation.isDraftSaving && annotation.draftSaved !== before.draftSaved,
        3000,
      );
      return verify(annotation, before, "autosave-flush");
    } catch (e) {
      console.warn("[draft] autosave flush 실패", e);
    }
  }

  return {
    saved: false,
    method: "no-method",
    draftId: annotation.draftId || null,
    detail: "draft 저장 메서드를 찾지 못했습니다.",
  };
}

/**
 * 저장이 실제로 반영됐는지 상태 변화로 확인한다.
 * draftId가 새로 생겼거나 draftSaved 타임스탬프가 갱신되면 성공으로 본다.
 */
function verify(annotation, before, method) {
  const draftId = annotation.draftId ?? 0;
  const changed =
    (draftId && draftId !== before.draftId) ||
    (annotation.draftSaved && annotation.draftSaved !== before.draftSaved);

  return {
    saved: Boolean(changed),
    method,
    draftId: draftId || null,
    ...(changed
      ? {}
      : {
        detail:
          "저장 호출은 됐지만 draftId/draftSaved가 변하지 않았습니다. " +
          "변경사항이 없어 서버가 무시했을 수 있습니다.",
      }),
  };
}