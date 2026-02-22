export default function FramePreview({ selection, onClose }) {
  if (!selection) return null;

  const { frame, distance } = selection;
  const fallbackName = frame.rgb_path ? frame.rgb_path.split("/").pop() : "";
  const imgUrl = frame.image_url || (fallbackName ? `/frames/${fallbackName}` : "");
  const title = frame.session_name ? `${frame.session_name} · Frame #${frame.sample_id}` : `Frame #${frame.sample_id}`;

  return (
    <div className="frame-preview">
      <div className="frame-preview-header">
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
