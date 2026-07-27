// web/libs/editor/src/components/BottomBar/GeoreferencingButton.jsx
import { useState, useCallback, useEffect } from "react";
import { observer } from "mobx-react";
//import { IconGlobe } from "@humansignal/icons";   // 적절한 아이콘 선택
import { Button } from "@humansignal/ui";
import { cn } from "../../utils/bem";
import { Modal } from "../../common/Modal/ModalPopup";
import { GeoMapView } from "../Common/GeoMapView";

const IconGlobe = (props) => (
    <svg {...props} width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="12" r="10" />
        <line x1="2" y1="12" x2="22" y2="12" />
        <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
    </svg>
);

export const GeoreferencingButton = observer(({ store }) => {
    const [loading, setLoading] = useState(false);
    const taskId = store.task?.id;
    const [isRunning, setIsRunning] = useState(false);

    // ── 지도 팝업 표시 ─────────────────────────────────────────
    const showMapModal = useCallback((taskId, geojson) => {
        Modal.modal({
            title: `Georeferencing 결과 — Task #${taskId}`,
            body: <GeoMapView geojson={geojson} />,
            style: { width: "80vw", maxWidth: 1100 },   // 넓은 모달
            okText: "닫기",
            closeOnClickOutside: true,
        });
    }, []);

    // 초기 상태 로드
    useEffect(() => {
        if (!taskId) return;
        fetch(`/api/tasks/${taskId}/georeferencing`)
            .then((r) => r.json())
            //.then((data) => setExcluded(data.exclude_from_export ?? false))
            .catch(console.error);
    }, [taskId]);

    const execute = useCallback(
        async () => {
            setIsRunning(true);

            try {
                const response = await fetch(`/api/tasks/${taskId}/georeferencing`, {
                    method: "GET",
                    headers: {
                        "Content-Type": "application/json",
                        "X-CSRFToken": document.cookie.match(/csrftoken=([^;]+)/)?.[1] ?? "",
                    },
                });

                if (!response.ok) {
                    const text = await response.text();
                    throw new Error(`HTTP ${response.status}`);
                }

                const result = await response.json();
                if (!result.geojson) {
                    throw new Error("응답에 geojson이 없습니다");
                }

                showMapModal(taskId, result.geojson)
            } catch (e) {
                console.error("Georeferencing error:", e);
                alert(`Georeferencing 실패: ${e.message}`);
            } finally {
                setIsRunning(false);
            }
        },
        [showMapModal],
    );

    const handleClick = useCallback(() => {
        if (isRunning) return;

        const taskId = store.task?.id;
        if (!taskId) {
            console.warn("Georeferencing: no task id");
            return;
        }

        // 확인 없이 바로 계산 + 팝업 (확인창 원하면 Modal.confirm으로 감싸기)
        execute(taskId);
    }, [store, isRunning, execute]);

    return (
        <Button
            type="text"
            size="small"
            look="string"
            variant="neutral"
            onClick={handleClick}
            disabled={isRunning}
            tooltip={isRunning ? "Georeferencing 계산..." : "Georeferencing"}
            className="aspect-square"
            leading={<IconGlobe />}
            aria-label="Georeferencing"
            data-testid="region-panel-georeferencing-button"
        />
    );
});