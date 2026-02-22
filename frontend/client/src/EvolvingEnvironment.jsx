import { useState, useEffect } from "react";
import EvolvingViewer from "./EvolvingViewer";
import EvolvingFramePreview from "./EvolvingFramePreview";
import EvolvingChatbot from "./EvolvingChatbot";

const TIME_STOPS = [
  { id: "nochair", label: "0 min", minutes: 0 },
  { id: "1chair", label: "5 min", minutes: 5 },
  { id: "2chair", label: "10 min", minutes: 10 },
];

function DetectionLegend({ scene }) {
  const [detections, setDetections] = useState([]);

  useEffect(() => {
    fetch("/scenes/detections.json")
      .then((r) => r.json())
      .then((data) => {
        setDetections(data.scenes[scene]?.objects || []);
      });
  }, [scene]);

  if (detections.length === 0) return (
    <div className="detection-legend">
      <div className="detection-legend-empty">No objects detected</div>
    </div>
  );

  return (
    <div className="detection-legend">
      {detections.map((obj) => (
        <div key={obj.id} className="detection-item">
          <span
            className="detection-swatch"
            style={{ background: `rgb(${obj.color.map(c => Math.round(c * 255)).join(",")})` }}
          />
          <span className="detection-label">{obj.label}</span>
          <span className="detection-conf">{(obj.confidence * 100).toFixed(0)}%</span>
          <span className="detection-dims">
            {obj.dimensions.width.toFixed(2)} x {obj.dimensions.height.toFixed(2)} x {obj.dimensions.depth.toFixed(2)}m
          </span>
        </div>
      ))}
    </div>
  );
}

export default function EvolvingEnvironment() {
  const [timeIdx, setTimeIdx] = useState(0);
  const scene = TIME_STOPS[timeIdx].id;
  const [selection, setSelection] = useState(null);

  const handleTimeChange = (idx) => {
    setTimeIdx(idx);
    setSelection(null);
  };

  return (
    <div className="page-layout">
      <div className="panel panel-left">
        <div className="panel-header">
          <span>3D Point Cloud</span>
          <div className="time-slider-wrapper">
            <div className="time-labels">
              {TIME_STOPS.map((t, i) => (
                <span
                  key={t.id}
                  className={`time-label ${i === timeIdx ? "active" : ""}`}
                  onClick={() => handleTimeChange(i)}
                >
                  {t.label}
                </span>
              ))}
            </div>
            <input
              type="range"
              min={0}
              max={TIME_STOPS.length - 1}
              step={1}
              value={timeIdx}
              onChange={(e) => handleTimeChange(Number(e.target.value))}
              className="time-slider"
            />
          </div>
        </div>
        <EvolvingViewer
          key={scene}
          scene={scene}
          onFrameSelect={setSelection}
          selectedFrame={selection?.frame}
        />
        <DetectionLegend scene={scene} />
        <EvolvingFramePreview
          selection={selection}
          scene={scene}
          onClose={() => setSelection(null)}
        />
      </div>
      <div className="panel panel-right">
        <div className="panel-header">AI Assistant</div>
        <EvolvingChatbot
          currentScene={scene}
          currentTimeIdx={timeIdx}
          timeStops={TIME_STOPS}
          selection={selection}
        />
      </div>
    </div>
  );
}
