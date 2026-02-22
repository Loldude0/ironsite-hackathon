import { useState, useRef, useEffect } from "react";
import OpenAI from "openai";

const OPENAI_KEY = import.meta.env.VITE_OPENAI_API_KEY;

function getFrameImageUrl(frame) {
  if (!frame) return null;
  if (frame.image_url) return frame.image_url;
  const fallbackName = frame.rgb_path ? frame.rgb_path.split("/").pop() : "";
  return fallbackName ? `/frames/${fallbackName}` : null;
}

async function imageUrlToBase64(url, { flipX = false } = {}) {
  const res = await fetch(url);
  const blob = await res.blob();
  const img = await createImageBitmap(blob);
  const canvas = document.createElement("canvas");
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext("2d");
  if (flipX) {
    ctx.translate(img.width, 0);
    ctx.scale(-1, 1);
  }
  ctx.drawImage(img, 0, 0);
  return canvas.toDataURL("image/jpeg", 0.85);
}

export default function Chatbot({ selection, getCanvasScreenshot }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  if (!OPENAI_KEY) {
    return (
      <div className="env-error">
        Missing <code>VITE_OPENAI_API_KEY</code> in .env file
      </div>
    );
  }

  const client = new OpenAI({
    apiKey: OPENAI_KEY,
    dangerouslyAllowBrowser: true,
  });

  const send = async () => {
    const text = input.trim();
    if (!text || loading) return;

    const userMsg = { role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const imageContents = [];

      const birdEyeBase64 = getCanvasScreenshot?.();
      if (birdEyeBase64) {
        imageContents.push({
          type: "text",
          text: "Bird's-eye view of the 3D reconstructed environment. The gold/yellow highlighted region marks the user's currently selected location:",
        });
        imageContents.push({
          type: "image_url",
          image_url: { url: birdEyeBase64 },
        });
      }

      const frame = selection?.frame;
      const frameUrl = getFrameImageUrl(frame);
      if (frameUrl) {
        const frameBase64 = await imageUrlToBase64(frameUrl, { flipX: true });
        imageContents.push({
          type: "text",
          text: `Camera frame from the selected location (Frame #${frame.sample_id}):`,
        });
        imageContents.push({
          type: "image_url",
          image_url: { url: frameBase64 },
        });
      }

      const hasImages = imageContents.length > 0;

      const systemMsg = {
        role: "system",
        content:
          "You are an AI assistant for a multi-agent localization system that reconstructs 3D environments from multiple robot/device scans. " +
          (hasImages
            ? "You are provided with a bird's-eye view of the full 3D reconstructed point cloud and, if available, a camera frame image from the selected location. " +
              "In the bird's-eye view, the user's currently selected location is highlighted in gold/yellow. Use this highlighted area to understand where the user is currently located within the overall environment. " +
              "Use both images to give spatial context: the bird's-eye view shows the overall environment layout and the user's position, and the camera frame shows the first-person perspective from that position. "
            : "") +
          "Answer the user's questions about the environment, spatial layout, objects, or localization. " +
          "Speak naturally about the scene. Do not mention implementation details like JSON, point clouds, or base64 unless asked. " +
          "Do not use special characters, em dashes, or LaTeX formatting. Keep your response clean and plain text only.",
      };

      const userContent = hasImages
        ? [{ type: "text", text }, ...imageContents]
        : text;

      const apiMessages = [systemMsg, { role: "user", content: userContent }];

      const res = await client.chat.completions.create({
        model: hasImages ? "gpt-5.2" : "gpt-5-mini",
        messages: apiMessages,
      });

      const reply = res.choices[0].message.content;
      setMessages((prev) => [...prev, { role: "assistant", content: reply }]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${err.message}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  return (
    <>
      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-hint">
            Click a point in the 3D scan, then ask a question.
            <br /><br />
            The AI will see both the <strong>bird's-eye view</strong> of the environment and the <strong>camera frame</strong> from your selected location.
          </div>
        )}
        {messages.map((m, i) => (
          <div
            key={i}
            className={`chat-bubble ${
              m.role === "user"
                ? "user"
                : m.content?.startsWith("Error:")
                  ? "error"
                  : "assistant"
            }`}
          >
            {typeof m.content === "string" ? m.content : JSON.stringify(m.content)}
          </div>
        ))}
        {loading && (
          <div className="typing-indicator">
            <span /><span /><span />
          </div>
        )}
        <div ref={endRef} />
      </div>
      <div className="chat-input-area">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKey}
          placeholder={
            selection
              ? "Ask about this area..."
              : "Select a point first, then ask..."
          }
          disabled={loading}
        />
        <button onClick={send} disabled={loading || !input.trim()}>
          Send
        </button>
      </div>
    </>
  );
}
