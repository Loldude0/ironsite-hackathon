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

Realtime update cadence and default YOLO thresholds are configured in [config/realtime_yolo_config.json](config/realtime_yolo_config.json).
Use `window.show` and `window.name` there to control the YOLO display window.

Viewer-specific optional arguments:
- `--point-size`
- `--max-points`

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
