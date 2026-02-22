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
python viewer_entry.py assets/sample.pcd --video assets/sample_yolo.mp4 --camera-config config/sample_camera_config.json --model yolo26n.pt --realtime-config config/realtime_yolo_config.json --ray-radius 0.08
```

You can also pass a folder containing multiple `.pcd` files as the first argument. In that mode, the viewer updates the displayed point cloud frame-by-frame at the interval configured in `point_cloud.update_interval_ms`.

```bash
python viewer_entry.py assets/converted_pcd --video assets/sample_yolo.mp4 --camera-config config/sample_camera_config.json --model yolo26n.pt --realtime-config config/realtime_yolo_config.json --ray-radius 0.08
```

Realtime update cadence and default YOLO thresholds are configured in [config/realtime_yolo_config.json](config/realtime_yolo_config.json).
Use `window.show` and `window.name` there to control the YOLO display window.
Use `point_cloud.update_interval_ms` and `point_cloud.loop` there to control folder playback.

Viewer-specific optional arguments:
- `--point-size`
- `--max-points`

### 4) Convert `.bag` LiDAR stream to multiple `.pcd` files
Use this to extract PointCloud2 frames from a ROS `.bag` into a folder:

```bash
python bag_to_pcd.py --bag data/lidar.bag --output-dir data/pcd --topic /velodyne_points
```

Optional arguments:
- `--every-n` save every Nth frame
- `--max-frames` stop after saving a number of frames
- `--topic` omit to auto-pick first PointCloud2 topic in the bag

### 5) Record iPhone LiDAR (ARVOS websocket) to `.pcd` + pose JSON

```bash
python arvos_server.py --port 9090 --output-dir data/arvos_capture
```

Output structure:
- `data/arvos_capture/pcd/*.pcd`
- `data/arvos_capture/pose/*.json`

Each pose JSON includes camera position/orientation fields when provided by stream payload (quaternion, Euler, rotation matrix, and raw pose payload).

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
