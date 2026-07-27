// web/libs/editor/src/components/Common/GeoreferencingButton.jsx
import { useState, useCallback } from "react";
import { observer } from "mobx-react";
import { getRoot } from "mobx-state-tree";
import { Button } from "@humansignal/ui";
import { IconGlobe } from "@humansignal/icons";  // 없으면 아래 fallback 아이콘 사용
import { GeoreferencingResultsModal } from "./GeoreferencingResultsModal";
import { saveDraftBeforeRequest } from "./saveDraft";

/**
 * @param {object} region - LSF region
 */
export const GeoreferencingButton = observer(({ region }) => {
    const [isRunning, setIsRunning] = useState(false);
    const [phase, setPhase] = useState(null);
    const [modal, setModal] = useState(null); // { visible, result, error, taskId }

    if (!region) return null;

    const handleClick = useCallback(async () => {
        if (!region) {
            console.warn("GeoreferencingButton: region prop이 없습니다");
            return;
        }
        setIsRunning(true);

        console.log("region:", region);
        const regionId = region.cleanId ?? region.id?.split("#")[0];
        const annotationId = region.annotation?.pk ?? region.annotation?.id;
        const root = getRoot(region);
        const taskId = root.task?.id;

        try {
            // 1단계: draft 저장 — 새로 그리거나 옮긴 region을 서버와 동기화
            setPhase("draft");
            const draftResult = await saveDraftBeforeRequest(region, root);
            if (!draftResult.saved) {
                console.warn("[georeferencing] draft 미저장:", draftResult);
            }

            // 2단계: georeferencing 검색
            setPhase("search");
            const response = await fetch(
                `/api/annotations/${encodeURIComponent(regionId)}/georeferencing?task_id=${taskId}`,
                {
                    method: "GET",
                    headers: {
                        "Content-Type": "application/json",
                        "X-CSRFToken": document.cookie.match(/csrftoken=([^;]+)/)?.[1] ?? "",
                    },
                },
            );

            const result = await response.json().catch(() => null);

            if (response.status === 409) {
                //alert(result?.detail ?? "지리 좌표 미계산 — DetectionRegion 백필 필요");
                setModal({
                    visible: true,
                    error: result?.detail ?? "지리 좌표 미계산 — DetectionRegion 백필 필요",
                    taskId,
                });
                return;
            }
            if (!response.ok) throw new Error(result?.detail ?? `HTTP ${response.status}`);

            // TODO: alert 대신 ResultsModal(이전에 전달한 컴포넌트)로 교체
            console.log("find-images result:", result);
            //alert(`이 영역(${annotationId})을 포함하는 이미지 ${result.count}개 발견 (현재 task #${taskId} 제외)`);
            const sourceCoords =
                result?.region?.coordinates ??      // 응답에 있으면 사용
                result?.source_region?.coordinates ??
                null;

            const labelValue =
                region.results?.[0]?.value?.rectanglelabels?.[0] ??
                region.results?.[0]?.value?.polygonlabels?.[0] ??
                region.results?.[0]?.value?.labels?.[0] ??
                "참조 영역";

            setModal({
                visible: true,
                result,
                taskId,
                sourceCoordinates: sourceCoords,
                sourceLabel: `${labelValue} (Task #${taskId})`,
            });
        } catch (e) {
            console.error("Georeferencing error:", e);
            //alert(`Georeferencing 실패: ${e.message}`);
            setModal({ visible: true, error: e.message, taskId });
        } finally {
            setIsRunning(false);
            setPhase(null);
        }
    }, [region, isRunning]);

    // Region 패널용 — 작은 아이콘 버튼
    return (
        <>
            <Button
                type="text"
                size="small"
                look="string"
                variant="neutral"
                onClick={handleClick}
                disabled={isRunning}
                tooltip={isRunning ? "Georeferencing 계산 중..." : "Georeferencing"}
                aria-label="Run georeferencing"
                leading={<IconGlobe style={{ width: 16, height: 16 }} />}
                data-testid="region-panel-georeferencing-button"
            />

            {modal?.visible && (
                <GeoreferencingResultsModal
                    result={modal.result}
                    error={modal.error}
                    currentTaskId={modal.taskId}
                    sourceCoordinates={modal.sourceCoordinates}
                    sourceLabel={modal.sourceLabel}
                    onHide={() => setModal(null)}
                />
            )}
        </>
    );
});