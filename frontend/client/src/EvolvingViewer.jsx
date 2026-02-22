import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls, Html } from "@react-three/drei";
import * as THREE from "three";

const HIGHLIGHT_RADIUS = 2.5;
const GOLD = [1.0, 0.84, 0.0];

function loadTrajectory(text) {
  const points = [];
  for (const line of text.split("\n")) {
    if (line.startsWith("#") || !line.trim()) continue;
    const p = line.trim().split(/\s+/);
    if (p.length >= 4) {
      points.push({ ts: parseFloat(p[0]), x: -parseFloat(p[1]), y: -parseFloat(p[2]), z: -parseFloat(p[3]) });
    }
  }
  return points;
}

function matchFramesToTrajectory(frames, trajPoints) {
  for (const f of frames) {
    let best = null;
    let bestDt = Infinity;
    for (const tp of trajPoints) {
      const dt = Math.abs(tp.ts - f.ts);
      if (dt < bestDt) {
        bestDt = dt;
        best = tp;
      }
    }
    if (best) {
      f.worldPos = [best.x, best.y, best.z];
    }
  }
}

function BoundingBox({ obj }) {
  const { vertices, edges } = obj.bbox;
  const [r, g, b] = obj.color;
  const color = new THREE.Color(r, g, b);

  const lineGeo = useMemo(() => {
    const points = [];
    for (const [a, b] of edges) {
      points.push(new THREE.Vector3(...vertices[a]));
      points.push(new THREE.Vector3(...vertices[b]));
    }
    const geo = new THREE.BufferGeometry().setFromPoints(points);
    return geo;
  }, [vertices, edges]);

  const fillGeo = useMemo(() => {
    const v = vertices;
    const faceIndices = [
      [0,1,2,3], [4,5,6,7],
      [0,1,5,4], [2,3,7,6],
      [0,3,7,4], [1,2,6,5],
    ];
    const pos = [];
    for (const [a, b, c, d] of faceIndices) {
      pos.push(...v[a], ...v[b], ...v[c]);
      pos.push(...v[a], ...v[c], ...v[d]);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    return geo;
  }, [vertices]);

  return (
    <group>
      <lineSegments geometry={lineGeo}>
        <lineBasicMaterial color={color} linewidth={2} />
      </lineSegments>
      <mesh geometry={fillGeo}>
        <meshBasicMaterial
          color={color}
          transparent
          opacity={0.08}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>
      <Html
        position={obj.center}
        center
        style={{ pointerEvents: "none" }}
      >
        <div style={{
          background: `rgba(${Math.round(r*255)}, ${Math.round(g*255)}, ${Math.round(b*255)}, 0.85)`,
          color: "#fff",
          padding: "3px 10px",
          borderRadius: "4px",
          fontSize: "11px",
          fontWeight: 600,
          fontFamily: "Inter, sans-serif",
          whiteSpace: "nowrap",
          letterSpacing: "0.3px",
          boxShadow: "0 2px 8px rgba(0,0,0,0.4)",
        }}>
          {obj.label}
          <span style={{ opacity: 0.7, marginLeft: 6, fontSize: "10px" }}>
            {(obj.confidence * 100).toFixed(0)}%
          </span>
        </div>
      </Html>
    </group>
  );
}

function DetectionOverlays({ detections }) {
  if (!detections || detections.length === 0) return null;
  return (
    <>
      {detections.map((obj) => (
        <BoundingBox key={obj.id} obj={obj} />
      ))}
    </>
  );
}

function PointCloud({ url, framesIndex, onPointClick, onMapCentroid, selectedFrame }) {
  const pointsRef = useRef();
  const baseColorsRef = useRef(null);
  const [geometry, setGeometry] = useState(null);
  const { camera, raycaster } = useThree();

  useEffect(() => {
    raycaster.params.Points.threshold = 0.15;
  }, [raycaster]);

  useEffect(() => {
    fetch(url)
      .then((r) => r.text())
      .then((text) => {
        const lines = text.split("\n");
        let headerEnd = 0;
        let vertexCount = 0;
        let hasColor = false;

        for (let i = 0; i < lines.length; i++) {
          if (lines[i].startsWith("element vertex")) {
            vertexCount = parseInt(lines[i].split(" ")[2], 10);
          }
          if (lines[i].startsWith("property") && lines[i].includes("red")) {
            hasColor = true;
          }
          if (lines[i].trim() === "end_header") {
            headerEnd = i + 1;
            break;
          }
        }

        const positions = new Float32Array(vertexCount * 3);
        const colors = new Float32Array(vertexCount * 3);
        let sx = 0, sy = 0, sz = 0;

        for (let i = 0; i < vertexCount; i++) {
          const line = lines[headerEnd + i];
          if (!line || !line.trim()) continue;
          const parts = line.trim().split(/\s+/);
          const x = -parseFloat(parts[0]);
          const y = -parseFloat(parts[1]);
          const z = -parseFloat(parts[2]);
          positions[i * 3] = x;
          positions[i * 3 + 1] = y;
          positions[i * 3 + 2] = z;
          sx += x; sy += y; sz += z;

          if (hasColor && parts.length >= 6) {
            colors[i * 3] = parseInt(parts[3], 10) / 255;
            colors[i * 3 + 1] = parseInt(parts[4], 10) / 255;
            colors[i * 3 + 2] = parseInt(parts[5], 10) / 255;
          } else {
            const norm = Math.sqrt(x * x + y * y + z * z);
            const t = Math.min(norm / 30, 1);
            colors[i * 3] = 0.2 + t * 0.5;
            colors[i * 3 + 1] = 0.6 - t * 0.2;
            colors[i * 3 + 2] = 0.9 - t * 0.4;
          }
        }

        const centroid = [sx / vertexCount, sy / vertexCount, sz / vertexCount];
        onMapCentroid(centroid);

        const geo = new THREE.BufferGeometry();
        geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
        geo.computeBoundingSphere();

        baseColorsRef.current = new Float32Array(colors);
        setGeometry(geo);

        if (geo.boundingSphere) {
          camera.position.set(14.04, 8.42, 7.98);
          camera.lookAt(10.19, 6.11, 5.79);
        }
      });
  }, [url, camera, onMapCentroid]);

  useEffect(() => {
    if (!geometry || !baseColorsRef.current) return;

    const colorAttr = geometry.getAttribute("color");
    const posAttr = geometry.getAttribute("position");
    const base = baseColorsRef.current;

    if (!selectedFrame) {
      for (let i = 0; i < colorAttr.count * 3; i++) {
        colorAttr.array[i] = base[i];
      }
      colorAttr.needsUpdate = true;
      return;
    }

    const [fx, fy, fz] = selectedFrame.worldPos;
    const r2 = HIGHLIGHT_RADIUS * HIGHLIGHT_RADIUS;

    for (let i = 0; i < posAttr.count; i++) {
      const dx = posAttr.array[i * 3] - fx;
      const dy = posAttr.array[i * 3 + 1] - fy;
      const dz = posAttr.array[i * 3 + 2] - fz;
      const dist2 = dx * dx + dy * dy + dz * dz;

      if (dist2 <= r2) {
        const fade = 1.0 - Math.sqrt(dist2) / HIGHLIGHT_RADIUS;
        const t = 0.5 + fade * 0.5;
        colorAttr.array[i * 3] = base[i * 3] * (1 - t) + GOLD[0] * t;
        colorAttr.array[i * 3 + 1] = base[i * 3 + 1] * (1 - t) + GOLD[1] * t;
        colorAttr.array[i * 3 + 2] = base[i * 3 + 2] * (1 - t) + GOLD[2] * t;
      } else {
        colorAttr.array[i * 3] = base[i * 3] * 0.4;
        colorAttr.array[i * 3 + 1] = base[i * 3 + 1] * 0.4;
        colorAttr.array[i * 3 + 2] = base[i * 3 + 2] * 0.4;
      }
    }
    colorAttr.needsUpdate = true;
  }, [selectedFrame, geometry]);

  const handleClick = useCallback(
    (e) => {
      e.stopPropagation();
      if (!framesIndex || framesIndex.length === 0) return;

      const point = e.point;
      let bestFrame = null;
      let bestDist = Infinity;

      for (const frame of framesIndex) {
        if (!frame.worldPos) continue;
        const dx = point.x - frame.worldPos[0];
        const dy = point.y - frame.worldPos[1];
        const dz = point.z - frame.worldPos[2];
        const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (d < bestDist) {
          bestDist = d;
          bestFrame = frame;
        }
      }

      if (bestFrame) {
        onPointClick({ frame: bestFrame, distance: bestDist });
      }
    },
    [framesIndex, onPointClick]
  );

  if (!geometry) return null;

  return (
    <points ref={pointsRef} geometry={geometry} onClick={handleClick}>
      <pointsMaterial vertexColors size={0.05} sizeAttenuation />
    </points>
  );
}

function CameraMarkers({ frames }) {
  if (!frames || frames.length === 0) return null;

  const geo = new THREE.BufferGeometry();
  const positions = new Float32Array(frames.length * 3);

  for (let i = 0; i < frames.length; i++) {
    if (!frames[i].worldPos) continue;
    positions[i * 3] = frames[i].worldPos[0];
    positions[i * 3 + 1] = frames[i].worldPos[1];
    positions[i * 3 + 2] = frames[i].worldPos[2];
  }

  geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));

  return (
    <points geometry={geo}>
      <pointsMaterial color="red" size={0.25} sizeAttenuation />
    </points>
  );
}

export default function PointCloudViewer({ scene, onFrameSelect, selectedFrame }) {
  const [framesIndex, setFramesIndex] = useState([]);
  const [mapCentroid, setMapCentroid] = useState(null);
  const [aligned, setAligned] = useState(false);
  const [detections, setDetections] = useState([]);

  useEffect(() => {
    Promise.all([
      fetch(`/scenes/${scene}/frames/samples_index.jsonl`).then((r) => r.text()),
      fetch(`/scenes/${scene}/trajectory_tum.txt`).then((r) => r.text()),
      fetch("/scenes/detections.json").then((r) => r.json()),
    ]).then(([samplesText, trajText, detectionsData]) => {
      const frames = samplesText
        .trim()
        .split("\n")
        .map((line) => JSON.parse(line));

      const trajPoints = loadTrajectory(trajText);
      matchFramesToTrajectory(frames, trajPoints);

      setFramesIndex(frames);
      setAligned(false);
      setMapCentroid(null);

      const sceneDetections = detectionsData.scenes[scene]?.objects || [];
      setDetections(sceneDetections);
    });
  }, [scene]);

  useEffect(() => {
    if (!mapCentroid || framesIndex.length === 0 || aligned) return;

    let sx = 0, sy = 0, sz = 0, n = 0;
    for (const f of framesIndex) {
      if (!f.worldPos) continue;
      sx += f.worldPos[0];
      sy += f.worldPos[1];
      sz += f.worldPos[2];
      n++;
    }
    if (n === 0) return;
    const camCentroid = [sx / n, sy / n, sz / n];

    const offset = [
      mapCentroid[0] - camCentroid[0],
      mapCentroid[1] - camCentroid[1],
      mapCentroid[2] - camCentroid[2],
    ];

    for (const f of framesIndex) {
      if (!f.worldPos) continue;
      f.worldPos = [
        f.worldPos[0] + offset[0],
        f.worldPos[1] + offset[1],
        f.worldPos[2] + offset[2],
      ];
    }

    setFramesIndex([...framesIndex]);
    setAligned(true);
  }, [mapCentroid, framesIndex, aligned]);

  const handleMapCentroid = useCallback((centroid) => {
    setMapCentroid(centroid);
  }, []);

  return (
    <Canvas
      className="viewer-canvas"
      camera={{ fov: 60, near: 0.1, far: 1000 }}
      gl={{ antialias: true }}
    >
      <ambientLight intensity={0.5} />
      <PointCloud
        url={`/scenes/${scene}/map_points.ply`}
        framesIndex={framesIndex}
        onPointClick={onFrameSelect}
        onMapCentroid={handleMapCentroid}
        selectedFrame={selectedFrame}
      />
      <DetectionOverlays detections={detections} />
      <CameraMarkers frames={framesIndex} />
      <OrbitControls enableDamping dampingFactor={0.1} rotateSpeed={0.8} />
      <gridHelper args={[50, 50, "#222", "#1a1a1a"]} position={[-15, 0, 0]} />
    </Canvas>
  );
}
