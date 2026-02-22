import { useState } from "react";
import "./App.css";
import PointCloudViewer from "./PointCloudViewer";
import FramePreview from "./FramePreview";
import Chatbot from "./Chatbot";

export default function App() {
  const [selection, setSelection] = useState(null);

  return (
    <div className="app-container">
      <div className="panel panel-left">
        <div className="panel-header">3D Point Cloud</div>
        <PointCloudViewer
          onFrameSelect={setSelection}
          selectedFrame={selection?.frame}
        />
        <FramePreview
          selection={selection}
          onClose={() => setSelection(null)}
        />
      </div>
      <div className="panel panel-right">
        <div className="panel-header">AI Assistant</div>
        <Chatbot />
      </div>
    </div>
  );
}
