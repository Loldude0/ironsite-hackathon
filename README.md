## Raycasting Visualizer

### Camera Config
Use `assets/sample_camera_config.json` (or your own JSON) to define:
- camera intrinsics (`fx`, `fy`, `cx`, `cy`, `width`, `height`)
- camera pose (`position`, `euler_deg`)
- visualization params (such as `image_plane_distance`)

### 1) Run viewer only (hardcoded sample boxes)
```bash
python -m pointcloud_locator.viewer assets/sample.pcd --camera-config assets/sample_camera_config.json --ray-radius 0.08
```

### 2) Run YOLO only (convert detections to viewer box format)
```bash
python yolo.py --image assets/sample_yolo.jpg --model yolo26n.pt
```

### 3) End-to-end: YOLO + Viewer (recommended)
This runs YOLO on an image, converts detections into the viewer-compatible bounding-box format, then launches the 3D viewer with those boxes.

If YOLO image size differs from `intrinsics.width/height` in camera config, the boxes are automatically rescaled to keep the floating image plane and box overlay aligned.

```bash
python viewer_entry.py assets/sample.pcd --image assets/sample_yolo.jpg --camera-config assets/sample_camera_config.json --model yolo26n.pt --ray-radius 0.08
```

Optional arguments (both scripts):
- `--conf` YOLO confidence threshold
- `--iou` YOLO IoU threshold
- `--device` inference device (e.g. `cpu` or `0`)

Viewer-specific optional arguments:
- `--point-size`
- `--max-points`

### Controls

- Left drag: rotate
- Right drag / Shift+Left drag: pan
- Mouse wheel: zoom
- R: reset view (viewer camera)
- Q or Esc: quit

Edit camera params:
- W/S A/D T/G : move +Z/-Z, -X/+X, +Y/-Y in camera local frame
- I/K J/L U/O : pitch+/-, yaw+/-, roll+/- (deg)
- Z/X         : decrease/increase fx, fy
- [ / ]       : move image plane nearer/farther
- , / .       : decrease/increase ray hit radius
- V           : toggle radius sphere visualization at each hit point
- H           : print current camera parameters
- P           : save current camera params to config file
