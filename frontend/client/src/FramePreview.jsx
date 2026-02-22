export default function FramePreview({ selection, onClose }) {
  if (!selection) return null;

  const { frame, distance } = selection;
  const filename = frame.rgb_path.split("/").pop();
  const imgUrl = `/frames/${filename}`;

  return (
    <div className="frame-preview">
      <div className="frame-preview-header">
        <span className="frame-preview-title">
          Frame #{frame.sample_id}
        </span>
        <span className="frame-preview-detail">{distance.toFixed(1)}m</span>
        <button className="frame-preview-close" onClick={onClose}>
          &times;
        </button>
      </div>
      <div className="frame-preview-img">
        <img src={imgUrl} alt={`Frame ${frame.sample_id}`} />
      </div>
    </div>
  );
}
