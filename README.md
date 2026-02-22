## Raycasting Visualizer

### Camera Config
Use `assets/sample_camera_config.json` (or your own JSON) to define:
- camera intrinsics (`fx`, `fy`, `cx`, `cy`, `width`, `height`)
- camera pose (`position`, `euler_deg`)
- visualization params (such as `image_plane_distance`)

### 1) Run viewer only (hardcoded sample boxes)
```bash
python -m pointcloud_locator.viewer assets/sample.pcd --camera-config config/sample_camera_config.json --ray-radius 0.08
```

### 2) Run YOLO only (single image -> viewer box format)
```bash
python yolo.py --image assets/sample_yolo.jpg --model yolo26n.pt
```

### 3) End-to-end realtime: MP4 + YOLO + Viewer (recommended)
This runs YOLO on a video stream, updates detections in realtime, and refreshes the floating 2D plane bounding boxes and raycast targets inside the viewer.
It also opens a separate YOLO video window showing annotated bounding boxes over the input video.

If YOLO image size differs from `intrinsics.width/height` in camera config, the boxes are automatically rescaled to keep the floating image plane and box overlay aligned.

```bash
python viewer_entry.py assets/sample.pcd --video assets/sample_yolo.mp4 --camera-config config/sample_camera_config.json --model yolo26n.pt --realtime-config config/realtime_yolo_config.json --device 0 --ray-radius 0.08
```

You can also pass a folder containing multiple `.pcd` files as the first argument. In that mode, the viewer updates the displayed point cloud frame-by-frame at the interval configured in `point_cloud.update_interval_ms`.

```bash
python viewer_entry.py assets/converted_pcd --video assets/big_buck_bunny.mp4 --camera-config config/sample_camera_config.json --model yolo26n.pt --realtime-config config/realtime_yolo_config.json --ray-radius 0.08
```

Realtime update cadence and default YOLO thresholds are configured in [config/realtime_yolo_config.json](config/realtime_yolo_config.json).
Use `--device` (for example `--device 0`) to force GPU, or set `yolo.device` in config. If neither is set, `viewer_entry.py` now auto-selects CUDA GPU `0` when available.
Use `window.show` and `window.name` there to control the YOLO display window.
Use `point_cloud.update_interval_ms` and `point_cloud.loop` there to control folder playback.

GPU prerequisites:
- Install a CUDA-enabled PyTorch build in your environment (Ultralytics uses PyTorch for GPU inference).
- Verify with:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-cuda')"
```

Viewer-specific optional arguments:
- `--point-size`
- `--max-points`

### 4) Convert `.bag` LiDAR stream to multiple `.pcd` files
Use this to extract PointCloud2 frames from a ROS `.bag` into a folder:

```bash
python bag_to_pcd.py --bag C:/Users/nishk/Downloads/dog_like.bag --output-dir assets/converted_pcd --max-frames 50
```

Optional arguments:
- `--every-n` save every Nth frame
- `--max-frames` stop after saving a number of frames
- `--topic` omit to auto-pick first PointCloud2 topic in the bag

### Controls

- Left drag: rotate
- Right drag / Shift+Left drag: pan
- Mouse wheel: zoom
- R: reset view (viewer camera)
- Q or Esc: quit
- In YOLO window, press `q` to stop the YOLO stream window

Edit camera params:
- W/S A/D T/G : move +Z/-Z, -X/+X, +Y/-Y in camera local frame
- I/K J/L U/O : pitch+/-, yaw+/-, roll+/- (deg)
- Z/X         : decrease/increase fx, fy
- [ / ]       : move image plane nearer/farther
- , / .       : decrease/increase ray hit radius
- V           : toggle radius sphere visualization at each hit point
- H           : print current camera parameters
- P           : save current camera params to config file

## Live iPhone Record3D USB -> ORB-SLAM3 (Tailscale)

This repo now includes a realtime RGB-D bridge:

- `windows_capture_record3d.py`: Record3D USB capture + validation
- `windows_stream_zmq.py`: ZMQ sender (`topic + JSON header + JPEG RGB + zstd depth`)
- `linux_orbslam3_rgbd_stream.cpp`: ZMQ receiver + `TrackRGBD`
- `configs/iphone_record3d_rgbd.yaml`: ORB-SLAM3 template (intrinsics filled from first frame)
- `scripts/run_linux_orbslam3.sh`: build + run on Linux
- `scripts/run_linux_xpra_viewer.sh`: run via xpra for remote Pangolin viewing

### Linux machine (CachyOS) startup

1. Ensure ORB-SLAM3 is built at `~/Projects/ORB_SLAM3` (or set `ORB_SLAM3_ROOT`).
2. Start receiver + ORB-SLAM3:

```bash
cd /home/atajne/Projects/ironsite-hackathon
chmod +x scripts/run_linux_orbslam3.sh scripts/run_linux_xpra_viewer.sh
./scripts/run_linux_orbslam3.sh
```

Default listener is `tcp://0.0.0.0:5555` on topic `rgbd`.
On `Ctrl+C`, the Linux bridge now saves:
- `outputs/trajectory_tum.txt`
- `outputs/keyframes_tum.txt`
- `outputs/map_points.ply`

Optional (viewer forwarded to Windows with xpra):

```bash
cd /home/atajne/Projects/ironsite-hackathon
./scripts/run_linux_xpra_viewer.sh
```

Then from Windows:

```powershell
xpra attach tcp:<linux_tailscale_ip>:14500
```

### Windows machine startup (after cloning this repo)

1. Install Python dependencies:

```powershell
cd <repo_path>
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Install Record3D Python package/SDK on Windows (from official Record3D docs).
3. Validate capture first:

```powershell
python windows_capture_record3d.py --preview --device-index 0 --depth-units auto --source-color-order RGB
```

4. Stream to Linux over Tailscale:

```powershell
python windows_stream_zmq.py --endpoint tcp://<linux_tailscale_ip>:5555 --device-index 0 --depth-units auto --source-color-order RGB --jpeg-quality 80 --zstd-level 3
```

For better realtime performance, downscale before streaming:

```powershell
python windows_stream_zmq.py --endpoint tcp://<linux_tailscale_ip>:5555 --device-index 0 --depth-units auto --source-color-order RGB --jpeg-quality 80 --zstd-level 3 --output-width 640 --output-height 480
```

To stream and also save per-frame artifacts (RGB/depth/meta + `pcd` per frame):

```powershell
python windows_stream_zmq.py --endpoint tcp://<linux_tailscale_ip>:5555 --device-index 0 --depth-units auto --source-color-order RGB --jpeg-quality 80 --zstd-level 3 --save-root .\recordings\session_01 --save-every-n 1 --save-pcd-stride 2
```

### Runtime notes

- If colors look wrong, switch `--source-color-order` between `RGB` and `BGR`.
- If scale is wrong, explicitly set `--depth-units meters` or `--depth-units millimeters`.
- First valid frame populates runtime ORB intrinsics from stream metadata.
- Linux bridge sensor mode can be switched with `SENSOR_MODE`:
  - RGB-D (default): `SENSOR_MODE=rgbd ./scripts/run_linux_orbslam3.sh`
  - RGB only (monocular): `SENSOR_MODE=monocular ./scripts/run_linux_orbslam3.sh`
- Frame sampling is enabled by default in the Linux bridge:
  - Every 15 frames it saves:
    - RGB image (`*_rgb.jpg`)
    - depth-derived point cloud (`*_depth.pcd`)
    - indexed pose metadata in `samples_index.jsonl` (`seq`, `ts`, `tracking_state`, `t_xyz_m`, `q_xyzw`, paths)
  - Output root: `outputs/frame_samples/run_<timestamp>_pid<id>/`
  - Configure with env vars:
    - `FRAME_SAMPLE_EVERY_N` (default `15`, set `0` to disable)
    - `FRAME_SAMPLE_DIR` (default `outputs/frame_samples`)
    - `FRAME_SAMPLE_PCD_PIXEL_STRIDE` (default `1`, use `2` or `4` for lighter files)

## UWB-Initialized Global Frame (Anchor + Worker v1)

This repo now includes a fusion service that estimates a fixed `T_G_Lw` from:
- anchor-side UWB samples from the iOS app (`IronsiteAIHack`)
- worker local SLAM poses streamed from `linux_orbslam3_rgbd_stream.cpp`

Supported modes:
- `paired_6dof`: UWB + SLAM paired within calibration window
- `initial_xyzyaw` (current app default): UWB-only 5s bootstrap, then lock `T_G_Lw` on first ORB pose using `x,y,z + yaw`

Outputs are written under `outputs/fusion/<session_id>/`:
- `calibration_result.json`
- `matched_samples.jsonl`
- `trajectory_global_tum.txt`

### Offline: transform `samples_index.jsonl` to global poses

If you already have sampled ORB poses and want to map them into the UWB/global frame offline:

```bash
python scripts/transform_samples_to_global.py \
  --samples-index outputs/frame_samples/run_<run_id>/samples_index.jsonl \
  --calibration-result outputs/fusion/<session_id>/calibration_result.json \
  --output-jsonl outputs/frame_samples/run_<run_id>/samples_global.jsonl \
  --output-tum outputs/frame_samples/run_<run_id>/trajectory_global_tum.txt
```

If calibration did not complete but you know one bootstrap global position+yaw (from UWB/image notes), you can build `T_G_Lw` from a single ORB sample:

```bash
python scripts/transform_samples_to_global.py \
  --samples-index outputs/frame_samples/run_<run_id>/samples_index.jsonl \
  --bootstrap \
  --global-xyz 1.2 -0.4 0.8 \
  --global-yaw-q-wxyz 0.96 0.0 0.0 0.28 \
  --bootstrap-sample-id 0 \
  --output-jsonl outputs/frame_samples/run_<run_id>/samples_global_bootstrap.jsonl \
  --write-transform-file outputs/frame_samples/run_<run_id>/T_G_Lw_bootstrap.json
```

Notes:
- The script expects sample quaternions in `q_xyzw` order.
- For yaw input, it supports `--global-yaw-rad`, `--global-yaw-deg`, `--global-yaw-q-xyzw`, or `--global-yaw-q-wxyz`.
- If your streamed ORB pose behaves like `Tcw` (world->camera), try `--invert-local-pose`.

### Linux startup (fusion + ORB bridge pose stream)

1. Install Python deps:
```bash
pip install -r requirements.txt
```

2. Start fusion service:
```bash
cd /home/atajne/Projects/ironsite-hackathon
./scripts/run_fusion_server.sh
```

3. Start ORB bridge with additive live pose output:
```bash
cd /home/atajne/Projects/ironsite-hackathon
SESSION_ID="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
POSE_STREAM_ENDPOINT=udp://127.0.0.1:8091 \
NODE_ID=worker_phone \
SESSION_ID="${SESSION_ID}" \
./scripts/run_linux_orbslam3.sh
```

Use the exact same session id in the iOS app `Session ID` field before pressing `Calibrate (5s)`.

### iOS app (`IronsiteAIHack`) flow

1. Set Fusion URL in the app UI: `http://<linux_tailscale_ip>:8080`
2. Set `Session ID` to match ORB `--session-id` / `SESSION_ID`.
3. On anchor phone, tap `Anchor (Host)`.
4. On worker phone, tap `Worker (Browse)`.
5. Tap `Calibrate (5s)` on anchor and move devices for ~5s.
6. In this mode, calibration enters `waiting_for_first_slam_pose` after UWB bootstrap is ready.
7. You can now switch worker phone to Record3D and start streaming; fusion finalizes on the first ORB pose.
8. Watch calibration state/countdown/matched pairs/inliers/confidence in the app.

The ORB tracking path is unchanged; live pose stream is additive via:
- `--pose-stream-endpoint`
- `--node-id`
- `--session-id`
- `--pose-stream-rate-hz`
