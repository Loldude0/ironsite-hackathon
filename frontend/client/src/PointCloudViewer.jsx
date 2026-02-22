import { useEffect, useRef, useState, useCallback } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";

const HIGHLIGHT_RADIUS = 2.5;
const GOLD = [1.0, 0.84, 0.0];

function parseJsonl(text) {
  const out = [];
  for (const line of text.split("\n")) {
    const raw = line.trim();
    if (!raw) continue;
    out.push(JSON.parse(raw));
  }
  return out;
}

function loadTrajectory(text) {
  const points = [];
  for (const line of text.split("\n")) {
    if (line.startsWith("#") || !line.trim()) continue;
    const p = line.trim().split(/\s+/);
    if (p.length >= 4) {
      points.push({ ts: parseFloat(p[0]), x: parseFloat(p[1]), y: parseFloat(p[2]), z: parseFloat(p[3]) });
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
      .then((r) => {
        if (!r.ok) {
          throw new Error(`Failed to load point cloud: ${url} (HTTP ${r.status})`);
        }
        return r.arrayBuffer();
      })
      .then((arrayBuffer) => {
        const loader = new PLYLoader();
        const geo = loader.parse(arrayBuffer);
        const posAttr = geo.getAttribute("position");
        if (!posAttr || posAttr.count <= 0) {
          throw new Error(`Invalid/empty PLY geometry for ${url}`);
        }

        const vertexCount = posAttr.count;
        const colors = new Float32Array(vertexCount * 3);
        let sx = 0;
        let sy = 0;
        let sz = 0;
        for (let i = 0; i < vertexCount; i++) {
          const x = posAttr.array[i * posAttr.itemSize];
          const y = posAttr.array[i * posAttr.itemSize + 1];
          const z = posAttr.array[i * posAttr.itemSize + 2];
          if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
            throw new Error(`PLY vertex contains invalid xyz at row ${i} for ${url}`);
          }
          sx += x;
          sy += y;
          sz += z;
        }

        const rawColorAttr = geo.getAttribute("color");
        if (rawColorAttr && rawColorAttr.count === vertexCount && rawColorAttr.itemSize >= 3) {
          let maxColor = 0.0;
          for (let i = 0; i < vertexCount; i++) {
            const r = rawColorAttr.array[i * rawColorAttr.itemSize];
            const g = rawColorAttr.array[i * rawColorAttr.itemSize + 1];
            const b = rawColorAttr.array[i * rawColorAttr.itemSize + 2];
            colors[i * 3] = r;
            colors[i * 3 + 1] = g;
            colors[i * 3 + 2] = b;
            maxColor = Math.max(maxColor, r, g, b);
          }
          if (maxColor > 1.0) {
            for (let i = 0; i < colors.length; i++) {
              colors[i] /= 255.0;
            }
          }
        } else {
          // Fallback gradient coloring when PLY has no color properties.
          for (let i = 0; i < vertexCount; i++) {
            const x = posAttr.array[i * posAttr.itemSize];
            const y = posAttr.array[i * posAttr.itemSize + 1];
            const z = posAttr.array[i * posAttr.itemSize + 2];
            const norm = Math.sqrt(x * x + y * y + z * z);
            const t = Math.min(norm / 30, 1);
            colors[i * 3] = 0.2 + t * 0.5;
            colors[i * 3 + 1] = 0.6 - t * 0.2;
            colors[i * 3 + 2] = 0.9 - t * 0.4;
          }
        }
        geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));

        const centroid = [sx / vertexCount, sy / vertexCount, sz / vertexCount];
        onMapCentroid(centroid);
        geo.computeBoundingSphere();

        baseColorsRef.current = new Float32Array(colors);
        setGeometry(geo);

        if (geo.boundingSphere) {
          const c = geo.boundingSphere.center;
          const r = geo.boundingSphere.radius;
          if (Number.isFinite(c.x) && Number.isFinite(c.y) && Number.isFinite(c.z) && Number.isFinite(r)) {
            camera.position.set(c.x + r, c.y + r, c.z + r);
            camera.lookAt(c.x, c.y, c.z);
          }
        }
      })
      .catch((err) => {
        console.error("[PointCloudViewer] point cloud load failed:", err);
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

export default function PointCloudViewer({ onFrameSelect, selectedFrame }) {
  const [framesIndex, setFramesIndex] = useState([]);
  const [mapCentroid, setMapCentroid] = useState(null);
  const [aligned, setAligned] = useState(false);
  const [usingPrecomputedWorld, setUsingPrecomputedWorld] = useState(false);

  useEffect(() => {
    const run = async () => {
      // Preferred path for manually-merged/demo scenes:
      // precomputed world positions and image URLs.
      try {
        const worldResp = await fetch("/frames/frames_world.jsonl", { cache: "no-store" });
        if (worldResp.ok) {
          const worldText = await worldResp.text();
          const framesWorld = parseJsonl(worldText);
          if (framesWorld.length > 0 && framesWorld.some((f) => Array.isArray(f.worldPos))) {
            setFramesIndex(framesWorld);
            setUsingPrecomputedWorld(true);
            setAligned(true);
            return;
          }
        }
      } catch {
        // Fall through to legacy path.
      }

      const [samplesText, trajText] = await Promise.all([
        fetch("/frames/samples_index.jsonl").then((r) => r.text()),
        fetch("/trajectory_tum.txt").then((r) => r.text()),
      ]);

      const frames = parseJsonl(samplesText);
      const trajPoints = loadTrajectory(trajText);
      matchFramesToTrajectory(frames, trajPoints);
      setFramesIndex(frames);
      setUsingPrecomputedWorld(false);
    };

    run();
  }, []);

  useEffect(() => {
    if (!mapCentroid || framesIndex.length === 0 || aligned || usingPrecomputedWorld) return;

    let sx = 0, sy = 0, sz = 0;
    for (const f of framesIndex) {
      if (!f.worldPos) continue;
      sx += f.worldPos[0];
      sy += f.worldPos[1];
      sz += f.worldPos[2];
    }
    const n = framesIndex.length;
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
  }, [mapCentroid, framesIndex, aligned, usingPrecomputedWorld]);

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
        url="/map_points.ply"
        framesIndex={framesIndex}
        onPointClick={onFrameSelect}
        onMapCentroid={handleMapCentroid}
        selectedFrame={selectedFrame}
      />
      <CameraMarkers frames={framesIndex} />
      <OrbitControls enableDamping dampingFactor={0.1} rotateSpeed={0.8} />
      <gridHelper args={[50, 50, "#222", "#1a1a1a"]} />
    </Canvas>
  );
}
