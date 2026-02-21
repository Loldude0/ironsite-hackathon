from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from pointcloud_locator import BoundingBox, yolo_output_to_viewer_bboxes


def predict_bboxes_for_viewer(
	image_path: str | Path,
	model_path: str | Path = "yolo26n.pt",
	conf: float | None = None,
	iou: float | None = None,
	device: str | None = None,
) -> list[BoundingBox]:
	"""Run YOLO on an image and return viewer-compatible bounding boxes."""
	boxes, _source_size = predict_bboxes_for_viewer_with_image_size(
		image_path=image_path,
		model_path=model_path,
		conf=conf,
		iou=iou,
		device=device,
	)
	return boxes


def predict_bboxes_for_viewer_with_image_size(
	image_path: str | Path,
	model_path: str | Path = "yolo26n.pt",
	conf: float | None = None,
	iou: float | None = None,
	device: str | None = None,
) -> tuple[list[BoundingBox], tuple[int, int] | None]:
	"""Run YOLO and return viewer boxes and source image size as ``(w, h)``."""
	model = YOLO(str(model_path))

	predict_kwargs: dict[str, object] = {
		"source": str(image_path),
	}
	if conf is not None:
		predict_kwargs["conf"] = conf
	if iou is not None:
		predict_kwargs["iou"] = iou
	if device is not None:
		predict_kwargs["device"] = device

	results = model.predict(**predict_kwargs)
	boxes = yolo_output_to_viewer_bboxes(results)

	source_size: tuple[int, int] | None = None
	if len(results) > 0 and hasattr(results[0], "orig_shape"):
		h, w = results[0].orig_shape[:2]
		source_size = (int(w), int(h))

	return boxes, source_size


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Run YOLO and emit viewer-compatible boxes.")
	parser.add_argument("--image", type=Path, default=Path("assets/sample_yolo.jpg"), help="Input image path")
	parser.add_argument("--model", type=Path, default=Path("yolo26n.pt"), help="YOLO model weights path")
	parser.add_argument("--conf", type=float, default=None, help="YOLO confidence threshold")
	parser.add_argument("--iou", type=float, default=None, help="YOLO IoU threshold")
	parser.add_argument("--device", type=str, default=None, help="Device, e.g. cpu or 0")
	return parser


def main() -> None:
	args = _build_parser().parse_args()
	viewer_bboxes = predict_bboxes_for_viewer(
		image_path=args.image,
		model_path=args.model,
		conf=args.conf,
		iou=args.iou,
		device=args.device,
	)
	print(viewer_bboxes)
	print(f"Converted {len(viewer_bboxes)} detections for viewer usage")


if __name__ == "__main__":
	main()
