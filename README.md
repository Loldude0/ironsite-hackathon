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

### Runtime notes

- If colors look wrong, switch `--source-color-order` between `RGB` and `BGR`.
- If scale is wrong, explicitly set `--depth-units meters` or `--depth-units millimeters`.
- First valid frame populates runtime ORB intrinsics from stream metadata.
- Linux bridge sensor mode can be switched with `SENSOR_MODE`:
  - RGB-D (default): `SENSOR_MODE=rgbd ./scripts/run_linux_orbslam3.sh`
  - RGB only (monocular): `SENSOR_MODE=monocular ./scripts/run_linux_orbslam3.sh`
