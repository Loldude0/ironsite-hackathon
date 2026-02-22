# AnchorVision: 4D Spatial Memory and UWB-Anchored SLAM for Construction

## Judge TL;DR

AnchorVision is a cross-platform system that transforms mobile video scans into a **globally anchored, queryable 4D construction memory**.

Traditional SLAM fails in construction because sites are GPS-denied, visually repetitive, and highly dynamic. AnchorVision solves this by combining:

1. **RGB-D SLAM** for high-fidelity local geometry.
2. **UWB (Ultra-Wideband) Ranging** for sparse global anchoring, eliminating the need for overlap-heavy loop closures across multi-agent sessions.
3. **Ray-casted 3D Semantics (YOLO26 + Depth)** to track objects and hazards across time.
4. **VLM-Powered Spatial Queries** to ask natural language questions about specific coordinates in the map (e.g., *"What changed in this hallway between 8 AM and 10 AM?"*).

This is not a toy pipeline. It is an end-to-end stack spanning iOS, Windows, Linux, and Web, backed by formal factor-graph estimation, robust calibration logic, and artifact-backed outputs.

---

## 🏗️ The Problem: Why Construction SLAM Breaks

Accurate, up-to-date 3D understanding is a prerequisite for construction progress tracking and safety auditing. However, construction sites routinely challenge traditional visual SLAM:

* **Repetitive corridors and low texture** cause severe tracking drift.
* **Dynamic occlusions** (moving workers, changing equipment) break map consistency.
* **Multi-agent alignment** requires workers to perfectly cross paths to establish loop closures, which is operationally unrealistic.

Teams don't just need a 3D map; they need a **spatiotemporal evidence index** that is geometrically reliable over time and queryable by non-roboticists.

---

## 🔬 Our Approach & Research Framing

Our core thesis is that sparse UWB constraints—already used in indoor positioning systems—can serve as lightweight global priors that reduce drift and loosen the operational constraints of collaborative SLAM.

### 1. The Estimation View (UWB-Anchored SLAM)

We model the multi-session fusion as a Maximum A Posteriori (MAP) optimization over visual, depth, and UWB constraints. We pull each agent's local SLAM trajectory into a shared site frame using UWB anchor distances:

This means global alignment **does not rely solely on visual loop closures**. UWB provides a consistent initialization and stabilization mechanism.

### 2. Spatiotemporal Semantic Lifting (4D Mapping)

We answer not only *what is the geometry*, but *what is where, and when*.

1. We run **YOLO26** on the RGB stream to detect objects.
2. We cast a ray from the camera center through the 2D bounding box and intersect it with the SLAM point cloud.
3. We transform this local coordinate into the UWB-anchored global frame, creating a semantic tuple: .

---

## 🚀 Product Features: What AnchorVision Actually Does

Our web frontend acts as a 4D spatial memory interface, bringing the research to life:

* **Cross-Session Spatial Retrieval:** Click anywhere on the fused 3D map to instantly pull up the exact RGB frame and timestamp for that physical location, pulling seamlessly from multiple independent worker sessions (`s1`, `s2`, `s3`).
* **VLM Scene Understanding:** Click a junction and ask the AI Assistant, *"What do you see in the area?"* The system feeds the spatially-indexed frame to a Vision-Language Model to generate rich architectural descriptions (e.g., *“polished concrete, exposed cable trays, drywall framing”*).
* **4D Timeline Visualization:** Use the "Evolving Environment" slider to scrub through time. By lifting 2D YOLO detections into 3D bounding boxes, the map visualizes exactly when and where objects (like equipment or chairs) appear, disappear, or move.
* **Spatially-Aware Change Analytics:** Select a specific 3D region and ask, *"How did this area change over time?"* The system cross-references the semantic index across temporal scans to provide a precise summary of object state changes, explicitly bounded to your queried geographic zone.
---

## 🛠️ Process & System Architecture

Building this required bridging mobile consumer hardware with edge-compute SLAM backends.

1. **Wearable/Agent (iPhone):** Captures RGB-D via Record3D and streams it over ZMQ. Simultaneously runs our custom iOS app leveraging Apple's Nearby Interaction (UWB) to stream ranges to an anchor.
2. **Edge Server (Linux/Windows):** * A realtime bridge (`linux_orbslam3_rgbd_stream.cpp`) ingests the ZMQ stream, runs ORB-SLAM3, and broadcasts a UDP pose stream.
* A Python fusion service (`fusion/solver.py`) takes the SLAM poses and UWB ranges, applying robust inlier gating (MAD-based rejection) to solve for the global site transform.


3. **Indexing & UI:** Post-processing scripts merge the sessions, project the 3D semantics, and build a highly optimized JSONL world index consumed by a React frontend.

---

## 📊 Research Findings & Artifact-Backed Evidence

All values below are extracted from artifacts present in this repository, proving our end-to-end integration works across multiple sessions.

| Metric | Value | Source / Notes |
| --- | --- | --- |
| **Indexed records written** | 146 | `frames_world.jsonl.summary.json` (Across 3 sessions) |
| **Filtered records (bad tracking)** | 1 | Proves our tracking-state gating works. |
| **Merged map vertices** | 46,599 | Header of `map_points.ply` |
| **Trajectories fused** | 3 (`s1, s2, s3`) | Spanning 2,215 total tracked keyframes. |
| **Solver unit tests** | 7 passed | `python -m unittest tests/test_fusion_solver.py` |

**Hypotheses Validated During Prototyping:**

* **H1 (Global Consistency):** UWB anchoring successfully placed three independent hallway traversals into a shared coordinate space without requiring heavy visual feature overlap.
* **H2 (Retrieval Utility):** World-indexed retrieval successfully mapped abstract 3D coordinates back to actionable visual evidence and accurate VLM context.

---

## 🏃 Reproduce the Demo

### Quickstart (Frontend & Product Demo)

```bash
cd frontend/client
npm install
npm run dev

```

The UI loads the pre-processed `/map_points.ply` and `/frames/frames_world.jsonl`.
*To enable the AI Assistant panel, set `VITE_OPENAI_API_KEY=...` in `frontend/client/.env`.*

*(For the full capture  fusion  indexing pipeline instructions, please see the `docs/PIPELINE.md` or the script execution order in the codebase).*

---

## 🔮 Limitations & Future Work

* **Quantitative Benchmarking:** While we achieved qualitative multi-session consistency, formal benchmarking (Chamfer distance, ATE/RPE, map-to-map ICP residuals) is planned for the post-hackathon phase.
* **UWB Degradation:** UWB is sensitive to Non-Line-of-Sight (NLoS) and human body shadowing. While our solver utilizes robust outlier rejection, advanced NLoS-aware error models are a necessary next step.