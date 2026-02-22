import { useState, useRef, useEffect } from "react";
import OpenAI from "openai";

const OPENAI_KEY = import.meta.env.VITE_OPENAI_API_KEY;

function getFrameImageUrl(scene, frame) {
  if (!frame) return null;
  const filename = frame.rgb_path.split("/").pop();
  return `/scenes/${scene}/frames/${filename}`;
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

async function findNearestFrame(scene, clickedWorldPos) {
  const res = await fetch(`/scenes/${scene}/frames/samples_index.jsonl`);
  const text = await res.text();
  const frames = text.trim().split("\n").map((l) => JSON.parse(l));

  const trajRes = await fetch(`/scenes/${scene}/trajectory_tum.txt`);
  const trajText = await trajRes.text();
  const trajPoints = [];
  for (const line of trajText.split("\n")) {
    if (line.startsWith("#") || !line.trim()) continue;
    const p = line.trim().split(/\s+/);
    if (p.length >= 4) {
      trajPoints.push({ ts: parseFloat(p[0]), x: -parseFloat(p[1]), y: -parseFloat(p[2]), z: -parseFloat(p[3]) });
    }
  }

  for (const f of frames) {
    let best = null, bestDt = Infinity;
    for (const tp of trajPoints) {
      const dt = Math.abs(tp.ts - f.ts);
      if (dt < bestDt) { bestDt = dt; best = tp; }
    }
    if (best) f.worldPos = [best.x, best.y, best.z];
  }

  if (!clickedWorldPos) return frames[Math.floor(frames.length / 2)];

  let bestFrame = frames[0], bestDist = Infinity;
  for (const f of frames) {
    if (!f.worldPos) continue;
    const dx = clickedWorldPos[0] - f.worldPos[0];
    const dy = clickedWorldPos[1] - f.worldPos[1];
    const dz = clickedWorldPos[2] - f.worldPos[2];
    const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (d < bestDist) { bestDist = d; bestFrame = f; }
  }
  return bestFrame;
}

async function getDetectionsForScene(scene) {
  const res = await fetch("/scenes/detections.json");
  const data = await res.json();
  return data.scenes[scene]?.objects || [];
}

function isPointNearBBox(point, obj, margin = 2.0) {
  if (!point || !obj.bbox?.vertices) return false;
  const verts = obj.bbox.vertices;
  const xs = verts.map((v) => v[0]);
  const ys = verts.map((v) => v[1]);
  const zs = verts.map((v) => v[2]);
  const minX = Math.min(...xs) - margin;
  const maxX = Math.max(...xs) + margin;
  const minY = Math.min(...ys) - margin;
  const maxY = Math.max(...ys) + margin;
  const minZ = Math.min(...zs) - margin;
  const maxZ = Math.max(...zs) + margin;
  return (
    point[0] >= minX && point[0] <= maxX &&
    point[1] >= minY && point[1] <= maxY &&
    point[2] >= minZ && point[2] <= maxZ
  );
}

function filterDetectionsNearPoint(objects, worldPos) {
  if (!worldPos) return [];
  return objects.filter((obj) => isPointNearBBox(worldPos, obj));
}

function formatDetections(objects) {
  if (objects.length === 0) return "No objects detected in the selected area.";
  return objects.map((o) =>
    `- ${o.label} (id: ${o.id}, confidence: ${(o.confidence * 100).toFixed(0)}%, ` +
    `dimensions: ${o.dimensions.width.toFixed(2)}m x ${o.dimensions.height.toFixed(2)}m x ${o.dimensions.depth.toFixed(2)}m, ` +
    `center: [${o.center.map(c => c.toFixed(2)).join(", ")}])`
  ).join("\n");
}

export default function Chatbot({ currentScene, currentTimeIdx, timeStops, selection }) {
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

  const numTimepoints = timeStops.length;
  const timeRange = `${timeStops[0].label} to ${timeStops[numTimepoints - 1].label}`;

  const classifyQuestion = async (question) => {
    const res = await client.chat.completions.create({
      model: "gpt-5-mini",
      messages: [
        {
          role: "system",
          content:
            "You classify user questions about a spatial environment that has been scanned at multiple points in time. " +
            "Respond with ONLY the word 'time' if the question involves changes over time, comparisons across different scans, temporal progression, what appeared, disappeared, or moved. " +
            "Respond with ONLY the word 'single' if the question can be answered from a single scan, e.g. identifying objects, describing the scene, spatial queries about the current view.",
        },
        { role: "user", content: question },
      ],
    });
    const answer = res.choices[0].message.content.trim().toLowerCase();
    return answer.includes("time") ? "time" : "single";
  };

  const buildTimeBasedMessages = async (question, clickedWorldPos) => {
    const imageContents = [];

    for (let i = 0; i < timeStops.length; i++) {
      const stop = timeStops[i];
      const frame = await findNearestFrame(stop.id, clickedWorldPos);
      const imgUrl = getFrameImageUrl(stop.id, frame);
      const base64 = await imageUrlToBase64(imgUrl, { flipX: true });
      const allDetections = await getDetectionsForScene(stop.id);
      const nearby = filterDetectionsNearPoint(allDetections, clickedWorldPos);

      imageContents.push({
        type: "text",
        text: `--- Scan ${i + 1} of ${timeStops.length}: ${stop.label} ---\nDetected objects in selected area:\n${formatDetections(nearby)}`,
      });
      imageContents.push({
        type: "image_url",
        image_url: { url: base64 },
      });
    }

    return [
      {
        role: "system",
        content:
          "You are an AI assistant analyzing a spatial environment that has been scanned at multiple points in time using 3D reconstruction. " +
          `You are provided ${numTimepoints} scans of the same area taken at different times (${timeRange}). ` +
          "Each scan includes a camera frame image and a list of detected objects with their bounding box dimensions, confidence scores, and 3D positions. " +
          "Your job is to analyze differences between scans, what objects appeared, disappeared, moved, or changed. " +
          "Be precise: reference specific objects, their positions, and which scans they appear in. " +
          "Do not mention implementation details like JSON, bounding boxes, or confidence scores unless the user asks. Speak naturally about the scene. " +
          "Do not use special characters, em dashes, or LaTeX formatting. Keep your response clean and plain text only.",
      },
      {
        role: "user",
        content: [
          { type: "text", text: question },
          ...imageContents,
        ],
      },
    ];
  };

  const buildSingleMessages = async (question, scene, clickedWorldPos) => {
    const frame = selection?.frame || await findNearestFrame(scene, clickedWorldPos);
    const imgUrl = getFrameImageUrl(scene, frame);
    const base64 = await imageUrlToBase64(imgUrl, { flipX: true });
    const allDetections = await getDetectionsForScene(scene);
    const nearby = filterDetectionsNearPoint(allDetections, clickedWorldPos);
    const currentStop = timeStops.find((t) => t.id === scene);

    return [
      {
        role: "system",
        content:
          "You are an AI assistant analyzing a spatial environment scanned using 3D reconstruction. " +
          `You are viewing scan: ${currentStop?.label}. ` +
          "You have a camera frame image and, if any objects were detected near the selected area, their details. " +
          "Answer the user's question about what is visible in the scene. " +
          "Do not mention implementation details like JSON, bounding boxes, or confidence scores unless the user asks. Speak naturally about the scene. " +
          "Do not use special characters, em dashes, or LaTeX formatting. Keep your response clean and plain text only.",
      },
      {
        role: "user",
        content: [
          {
            type: "text",
            text: `${question}\n\nDetected objects in selected area:\n${formatDetections(nearby)}`,
          },
          {
            type: "image_url",
            image_url: { url: base64 },
          },
        ],
      },
    ];
  };

  const send = async () => {
    const text = input.trim();
    if (!text || loading) return;

    const userMsg = { role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const clickedWorldPos = selection?.frame?.worldPos || null;
      const questionType = await classifyQuestion(text);

      let apiMessages;
      if (questionType === "time") {
        apiMessages = await buildTimeBasedMessages(text, clickedWorldPos);
      } else {
        apiMessages = await buildSingleMessages(text, currentScene, clickedWorldPos);
      }

      const res = await client.chat.completions.create({
        model: "gpt-5.2",
        messages: apiMessages,
      });

      const reply = res.choices[0].message.content;
      const tag = questionType === "time"
        ? `Compared ${numTimepoints} scans`
        : `Scan: ${timeStops[currentTimeIdx]?.label}`;

      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: reply, tag },
      ]);
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
            Try temporal queries like <em>"What changed?"</em> or spatial queries like <em>"What's in this area?"</em>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`chat-bubble ${m.role === "user" ? "user" : m.content?.startsWith("Error:") ? "error" : "assistant"}`}>
            {m.tag && <div className="chat-tag">{m.tag}</div>}
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
          placeholder={selection ? "Ask about this area..." : "Select a point first, then ask..."}
          disabled={loading}
        />
        <button onClick={send} disabled={loading || !input.trim()}>
          Send
        </button>
      </div>
    </>
  );
}
