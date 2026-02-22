import { useState, useRef, useCallback } from "react";
import LocalizationViewer from "./LocalizationViewer";
import LocalizationFramePreview from "./LocalizationFramePreview";
import LocalizationChatbot from "./LocalizationChatbot";

export default function MultiAgentLocalization() {
  const [selection, setSelection] = useState(null);
  const screenshotFnRef = useRef(null);

  const handleCanvasReady = useCallback((fn) => {
    screenshotFnRef.current = fn;
  }, []);

  const getCanvasScreenshot = useCallback(() => {
    if (screenshotFnRef.current) {
      return screenshotFnRef.current();
    }
    return null;
  }, []);

  return (
    <div className="page-layout">
      <div className="panel panel-left">
        <div className="panel-header">3D Point Cloud</div>
        <LocalizationViewer
          onFrameSelect={setSelection}
          selectedFrame={selection?.frame}
          onCanvasReady={handleCanvasReady}
        />
        <LocalizationFramePreview
          selection={selection}
          onClose={() => setSelection(null)}
        />
      </div>
      <div className="panel panel-right">
        <div className="panel-header">AI Assistant</div>
        <LocalizationChatbot
          selection={selection}
          getCanvasScreenshot={getCanvasScreenshot}
        />
      </div>
    </div>
  );
}
