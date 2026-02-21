## Raycasting Visualizer
### Camera Config
Edit the `config/camera_config.json` file to set the camera intrinsics and extrinsics. The intrinsics should be in the format of a 3x3 matrix, and the extrinsics should be in the format of a 4x4 matrix.

### Running the Visualizer
To run the visualizer, use the following command:
```bash
python -m pointcloud_locator.viewer assets/sample.pcd --camera-config assets/sample_camera_config.json --ray-radius 0.08
```
This command will load the point cloud from `assets/sample.pcd`, use the camera configuration from `assets/sample_camera_config.json`, and set the ray radius to 0.08 for better visualization. You can adjust the ray radius as needed for different point cloud densities.

### Controls

Left drag: rotate
Right drag / Shift+Left drag: pan
Mouse wheel: zoom
R: reset view (viewer camera)
Q or Esc: quit
Edit camera params:
    W/S A/D T/G : move +Z/-Z, -X/+X, +Y/-Y in camera local frame
    I/K J/L U/O : pitch+/-, yaw+/-, roll+/- (deg)
    Z/X         : decrease/increase fx, fy
    [ / ]       : move image plane nearer/farther
    , / .       : decrease/increase ray hit radius
    V           : toggle radius sphere visualization at each hit point
    H           : print current camera parameters
