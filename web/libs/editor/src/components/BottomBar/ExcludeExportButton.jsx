// web/libs/editor/src/components/BottomBar/ExcludeExportButton.jsx (신규)
import { useState, useCallback, useEffect } from "react";
import { observer } from "mobx-react";
import { IconBan } from "@humansignal/icons";   // 적절한 아이콘 선택
import { Button } from "@humansignal/ui";

export const ExcludeExportButton = observer(({ store }) => {
    const [excluded, setExcluded] = useState(false);
    const [loading, setLoading] = useState(false);
    const taskId = store.task?.id;

    // 초기 상태 로드
    useEffect(() => {
        if (!taskId) return;
        fetch(`/api/tasks/${taskId}`)
            .then((r) => r.json())
            .then((data) => setExcluded(data.exclude_from_export ?? false))
            .catch(console.error);
    }, [taskId]);

    const toggle = useCallback(async () => {
        if (!taskId || loading) return;
        setLoading(true);

        try {
            const response = await fetch(`/api/tasks/${taskId}`, {
                method: "PATCH",
                headers: {
                    "Content-Type": "application/json",
                    "X-CSRFToken": document.cookie.match(/csrftoken=([^;]+)/)?.[1] ?? "",
                },
                body: JSON.stringify({ exclude_from_export: !excluded }),
            });

            if (!response.ok) throw new Error(`HTTP ${response.status}`);

            setExcluded(!excluded);
        } catch (e) {
            console.error("Toggle exclude error:", e);
            alert(`변경 실패: ${e.message}`);
        } finally {
            setLoading(false);
        }
    }, [taskId, excluded, loading]);

    return (
        <Button
            type="text"
            size="small"
            look="string"
            variant={excluded ? "negative" : "neutral"}
            onClick={toggle}
            disabled={loading}
            tooltip={excluded ? "Export 제외됨 (클릭하여 포함)" : "Export에서 제외"}
            className="aspect-square"
            leading={<IconBan />}
            aria-label="Exclude from export"
            data-testid="bottombar-exclude-export-button"
        />
    );
});