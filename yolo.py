from __future__ import annotations

import argparse
import threading
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from pointcloud_locator import BoundingBox, yolo_output_to_viewer_bboxes


class RealtimeYoloVideoBBoxStream:
	"""Background YOLO stream that continuously publishes latest frame boxes."""

	def __init__(
		self,
		video_path: str | Path,
		model_path: str | Path = "yolo26n.pt",
		conf: float | None = None,
		iou: float | None = None,
		device: str | None = None,
		show_window: bool = False,
		window_name: str = "YOLO Realtime",
	) -> None:
		self.video_path = Path(video_path)
		self.model_path = Path(model_path)
		self.conf = conf
		self.iou = iou
		self.device = device
		self.show_window = show_window
		self.window_name = window_name

		self._lock = threading.Lock()
		self._stop_event = threading.Event()
		self._thread: threading.Thread | None = None

		self._latest_boxes: list[BoundingBox] = []
		self._latest_size: tuple[int, int] | None = None
		self._latest_seq: int = -1
		self._last_error: Exception | None = None

	def start(self) -> None:
		if self._thread is not None:
			return
		self._thread = threading.Thread(target=self._run, name="yolo-video-stream", daemon=True)
		self._thread.start()

	def stop(self, timeout: float = 2.0) -> None:
		self._stop_event.set()
		if self._thread is not None:
			self._thread.join(timeout=timeout)

	def get_latest(self) -> tuple[list[BoundingBox], tuple[int, int] | None, int]:
		with self._lock:
			return list(self._latest_boxes), self._latest_size, self._latest_seq

	def get_last_error(self) -> Exception | None:
		with self._lock:
			return self._last_error

	def _run(self) -> None:
		cv2 = None
		if self.show_window:
			try:
				import cv2 as _cv2
				cv2 = _cv2
			except Exception as exc:
				with self._lock:
					self._last_error = RuntimeError(f"OpenCV is required for YOLO display window: {exc}")
				return

		try:
			model = YOLO(str(self.model_path))

			predict_kwargs: dict[str, Any] = {
				"source": str(self.video_path),
				"stream": True,
				"verbose": False,
			}
			if self.conf is not None:
				predict_kwargs["conf"] = self.conf
			if self.iou is not None:
				predict_kwargs["iou"] = self.iou
			if self.device is not None:
				predict_kwargs["device"] = self.device

			for result in model.predict(**predict_kwargs):
				if self._stop_event.is_set():
					break

				boxes = yolo_output_to_viewer_bboxes(result)
				source_size: tuple[int, int] | None = None
				if hasattr(result, "orig_shape"):
					h, w = result.orig_shape[:2]
					source_size = (int(w), int(h))

				with self._lock:
					self._latest_boxes = boxes
					self._latest_size = source_size
					self._latest_seq += 1

				if cv2 is not None:
					annotated = result.plot()
					cv2.imshow(self.window_name, annotated)
					key = cv2.waitKey(1) & 0xFF
					if key == ord("q"):
						self._stop_event.set()
						break
		except Exception as exc:
			with self._lock:
				self._last_error = exc
		finally:
			if cv2 is not None:
				try:
					cv2.destroyWindow(self.window_name)
				except Exception:
					pass


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
