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
