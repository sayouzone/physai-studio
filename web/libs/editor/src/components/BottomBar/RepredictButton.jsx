// web/libs/editor/src/components/BottomBar/RepredictButton.jsx
import { useState, useCallback } from "react";
import { observer } from "mobx-react";
import { IconPredictions } from "@humansignal/icons";
import { Button } from "@humansignal/ui";
import { cn } from "../../utils/bem";
// ✅ Modal 경로는 프로젝트 구조에 맞게 조정
import { Modal } from "../../common/Modal/ModalPopup";

// ── Dialog 본문 컴포넌트 ──────────────────────────────────────────
const RepredictDialogContent = ({ taskId, modelVersion }) => {
    return (
        <div className={cn("dialog-content").toClassName()}>
            <div className={cn("dialog-content").elem("text").toClassName()}>
                현재 태스크 <strong>#{taskId}</strong>에 대해 재예측을 진행하시겠습니까?
                <br />
                <br />
                ML Backend{modelVersion ? ` (${modelVersion})` : ""}가 새로운 예측을 생성하며,
                기존 예측 결과가 갱신됩니다.
            </div>
        </div>
    );
};

// ── 재예측 버튼 ───────────────────────────────────────────────────
export const RepredictButton = observer(({ store }) => {
    const [isRepredicting, setIsRepredicting] = useState(false);

    // 실제 재예측 실행
    const executeRepredict = useCallback(
        async (taskId, projectId) => {
            setIsRepredicting(true);
            try {
                const response = await fetch(
                    `/api/dm/actions?id=retrieve_tasks_predictions&project=${projectId}`,
                    {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json",
                            "X-CSRFToken": document.cookie.match(/csrftoken=([^;]+)/)?.[1] ?? "",
                        },
                        body: JSON.stringify({
                            selectedItems: { all: false, included: [taskId] },
                        }),
                    },
                );

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

    // 확인 Dialog 표시
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

        const modelVersion = store.project?.model_version ?? null;

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
        <Button
            type="text"
            aria-label="재예측"
            size="small"
            look="string"
            variant="neutral"
            onClick={handleRepredict}
            disabled={isRepredicting}
            tooltip={isRepredicting ? "재예측 중..." : "재예측"}
            className="aspect-square"
            leading={<IconPredictions />}
            data-testid="bottombar-repredict-button"
        />
    );
});