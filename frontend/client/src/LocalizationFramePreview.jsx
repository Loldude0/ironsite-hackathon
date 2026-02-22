import { useState, useRef, useCallback } from "react";

export default function FramePreview({ selection, onClose }) {
  const [pos, setPos] = useState({ x: 20, y: 20 });
  const dragRef = useRef(null);

  const onMouseDown = useCallback((e) => {
    e.preventDefault();
    const startX = e.clientX;
    const startY = e.clientY;
    const startPos = { ...pos };

    const onMouseMove = (e) => {
      setPos({
        x: startPos.x + (e.clientX - startX),
        y: startPos.y - (e.clientY - startY),
      });
    };

    const onMouseUp = () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
    };

    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
  }, [pos]);

  if (!selection) return null;

  const { frame, distance } = selection;
  const fallbackName = frame.rgb_path ? frame.rgb_path.split("/").pop() : "";
  const imgUrl = frame.image_url || (fallbackName ? `/frames/${fallbackName}` : "");
  const title = frame.session_name
    ? `${frame.session_name} · Frame #${frame.sample_id}`
    : `Frame #${frame.sample_id}`;

  return (
    <div
      ref={dragRef}
      className="frame-preview frame-preview-draggable"
      style={{ left: pos.x, bottom: pos.y, right: "auto" }}
    >
      <div
        className="frame-preview-header frame-preview-drag-handle"
        onMouseDown={onMouseDown}
      >
        <span className="frame-preview-title">{title}</span>
        <span className="frame-preview-detail">{distance.toFixed(1)}m</span>
        <button className="frame-preview-close" onClick={onClose}>
          &times;
        </button>
      </div>
      <div className="frame-preview-img">
        <img src={imgUrl} alt={title} />
      </div>
    </div>
  );
}
