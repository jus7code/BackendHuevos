from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("huevops")
from contextlib import asynccontextmanager
from app.training import router as training_router, training_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await asyncio.to_thread(training_manager.shutdown)


app = FastAPI(title="HuevoPS Backend", version="0.1.0", lifespan=lifespan)
app.include_router(training_router)
VIDEO_MAX_FPS = 5.0
VIDEO_IMAGE_SIZE = 416


@dataclass
class Detection:
    id: str
    label: str
    status: str
    confidence: float
    box: list[float]


class IncrementalByteStream(io.RawIOBase):
    def __init__(self) -> None:
        self._chunks: queue.Queue[bytes | None] = queue.Queue()
        self._buffer = bytearray()
        self._closed = False
        self._closing = False

    def write(self, data: bytes) -> None:
        if not self._closed and not self._closing:
            self._chunks.put(data)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def close(self) -> None:
        if not self._closing:
            self._closing = True
            self._chunks.put(None)

    def read(self, size: int = -1) -> bytes:
        while not self._buffer and not self._closed:
            chunk = self._chunks.get()
            if chunk is None:
                self._closed = True
            else:
                self._buffer.extend(chunk)
        if size < 0 or size >= len(self._buffer):
            result = bytes(self._buffer)
            self._buffer.clear()
            return result
        result = bytes(self._buffer[:size])
        del self._buffer[:size]
        return result


class EggDetector:
    def __init__(self) -> None:
        model_path = Path(__file__).resolve().parent / "modelo_nuevo" / "best (1).pt"
        if not model_path.is_file():
            raise FileNotFoundError(f"No se encontro el modelo de predicciones: {model_path}")
        self._model = YOLO(str(model_path))
        self._model_lock = threading.Lock()
        logger.info("Modelo YOLO activo: %s; clases: %s", model_path, self._model.names)

    def detect(self, image: Image.Image, *, imgsz: int = 640) -> list[Detection]:
        # El modelo se comparte entre las peticiones HTTP y las sesiones de video.
        with self._model_lock:
            return self._detect_with_model(image, imgsz=imgsz)

    def _detect_with_onnx(self, image: Image.Image) -> list[Detection]:
        assert self._onnx_session is not None and self._onnx_input_name is not None
        original_width, original_height = image.size
        input_shape = self._onnx_session.get_inputs()[0].shape
        input_height = int(input_shape[2])
        input_width = int(input_shape[3])
        scale = min(input_width / original_width, input_height / original_height)
        resized_width = max(1, round(original_width * scale))
        resized_height = max(1, round(original_height * scale))
        resized = image.convert("RGB").resize((resized_width, resized_height), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (input_width, input_height), (114, 114, 114))
        pad_x = (input_width - resized_width) // 2
        pad_y = (input_height - resized_height) // 2
        canvas.paste(resized, (pad_x, pad_y))
        tensor = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        output = self._onnx_session.run(None, {self._onnx_input_name: tensor})[0]
        predictions = np.asarray(output)[0]
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.transpose(1, 0)

        candidates: list[tuple[float, list[float], str]] = []
        for prediction in predictions:
            if prediction.shape[0] < 5:
                continue
            confidence = float(prediction[4])
            if confidence < 0.25:
                continue
            center_x, center_y, box_width, box_height = map(float, prediction[:4])
            left = (center_x - box_width / 2 - pad_x) / scale
            top = (center_y - box_height / 2 - pad_y) / scale
            right = (center_x + box_width / 2 - pad_x) / scale
            bottom = (center_y + box_height / 2 - pad_y) / scale
            left = max(0.0, min(left, original_width))
            top = max(0.0, min(top, original_height))
            right = max(0.0, min(right, original_width))
            bottom = max(0.0, min(bottom, original_height))
            if right > left and bottom > top:
                candidates.append((confidence, [left, top, right, bottom], "egg"))

        detections: list[Detection] = []
        for index, (confidence, box, _) in enumerate(self._nms(candidates)):
            left, top, right, bottom = box
            status, _ = self._classify_basic(image, (int(left), int(top), int(right), int(bottom)))
            detections.append(
                Detection(
                    id=f"egg-{index + 1}",
                    label="Huevo",
                    status=status,
                    confidence=confidence,
                    box=[left / original_width, top / original_height, (right - left) / original_width, (bottom - top) / original_height],
                )
            )
        return detections[:500]

    @staticmethod
    def _nms(candidates: list[tuple[float, list[float], str]]) -> list[tuple[float, list[float], str]]:
        remaining = sorted(candidates, key=lambda candidate: candidate[0], reverse=True)
        selected: list[tuple[float, list[float], str]] = []
        while remaining:
            current = remaining.pop(0)
            selected.append(current)
            current_box = current[1]
            filtered: list[tuple[float, list[float], str]] = []
            for candidate in remaining:
                candidate_box = candidate[1]
                intersection_left = max(current_box[0], candidate_box[0])
                intersection_top = max(current_box[1], candidate_box[1])
                intersection_right = min(current_box[2], candidate_box[2])
                intersection_bottom = min(current_box[3], candidate_box[3])
                intersection = max(0.0, intersection_right - intersection_left) * max(0.0, intersection_bottom - intersection_top)
                current_area = max(0.0, current_box[2] - current_box[0]) * max(0.0, current_box[3] - current_box[1])
                candidate_area = max(0.0, candidate_box[2] - candidate_box[0]) * max(0.0, candidate_box[3] - candidate_box[1])
                union = current_area + candidate_area - intersection
                if union == 0.0 or intersection / union <= 0.45:
                    filtered.append(candidate)
            remaining = filtered
        return selected

    def _detect_with_model(self, image: Image.Image, *, imgsz: int = 640) -> list[Detection]:
        result = self._model.predict(image.convert("RGB"), imgsz=imgsz, verbose=False)[0]
        width, height = image.size
        detections: list[Detection] = []
        for index, (xyxy, confidence, class_id) in enumerate(
            zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist())
        ):
            x1, y1, x2, y2 = xyxy
            x1 = max(0.0, min(float(x1), width))
            y1 = max(0.0, min(float(y1), height))
            x2 = max(0.0, min(float(x2), width))
            y2 = max(0.0, min(float(y2), height))
            if x2 <= x1 or y2 <= y1:
                continue
            status = self._egg_status(str(result.names.get(int(class_id), "")))
            detections.append(
                Detection(
                    id=f"egg-{index + 1}",
                    label="Huevo",
                    status=status,
                    confidence=max(0.0, min(float(confidence), 1.0)),
                    box=[x1 / width, y1 / height, (x2 - x1) / width, (y2 - y1) / height],
                )
            )
        return detections[:500]

    @staticmethod
    def _egg_status(model_label: str) -> str:
        normalized = model_label.strip().lower()
        if any(word in normalized for word in ("roto", "broken", "cracked", "crack")):
            return "Roto"
        if any(word in normalized for word in ("sano", "healthy", "whole", "intacto")):
            return "Sano"
        return "Sano"

    @staticmethod
    def _classify_basic(image: Image.Image, box: tuple[int, int, int, int]) -> tuple[str, float]:
        x1, y1, x2, y2 = box
        margin_x = max(1, int((x2 - x1) * 0.15))
        margin_y = max(1, int((y2 - y1) * 0.15))
        crop_box = (x1 + margin_x, y1 + margin_y, x2 - margin_x, y2 - margin_y)
        if crop_box[2] < crop_box[0] or crop_box[3] < crop_box[1]:
            return "Sano", 0.5
        crop = np.asarray(image.convert("RGB").crop(crop_box))
        if crop.size == 0:
            return "Sano", 0.5
        brightness = crop.mean(axis=2)
        saturation = crop.max(axis=2) - crop.min(axis=2)
        dark_pixels = (brightness < 105) & (saturation < 85)
        dark_ratio = float(dark_pixels.mean())
        if dark_ratio > 0.035:
            confidence = min(0.98, 0.55 + dark_ratio * 3.0)
            return "Roto", confidence
        return "Sano", min(0.98, 0.75 + max(0.0, 0.035 - dark_ratio))

    def _detect_basic(self, image: Image.Image) -> list[Detection]:
        array = np.asarray(image.convert("RGB"))
        brightness = array.mean(axis=2)
        saturation = array.max(axis=2) - array.min(axis=2)
        mask = (brightness > 150) & (saturation < 95)
        height, width = mask.shape
        visited = np.zeros(mask.shape, dtype=bool)
        detections: list[Detection] = []
        minimum_area = max(64, int(width * height * 0.002))

        for y in range(0, height, 2):
            for x in range(0, width, 2):
                if not mask[y, x] or visited[y, x]:
                    continue
                stack = [(y, x)]
                visited[y, x] = True
                points: list[tuple[int, int]] = []
                while stack:
                    current_y, current_x = stack.pop()
                    points.append((current_y, current_x))
                    for next_y, next_x in (
                        (current_y - 2, current_x),
                        (current_y + 2, current_x),
                        (current_y, current_x - 2),
                        (current_y, current_x + 2),
                    ):
                        if 0 <= next_y < height and 0 <= next_x < width and mask[next_y, next_x] and not visited[next_y, next_x]:
                            visited[next_y, next_x] = True
                            stack.append((next_y, next_x))
                if len(points) < minimum_area:
                    continue
                ys = [point[0] for point in points]
                xs = [point[1] for point in points]
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)
                box_width, box_height = x2 - x1 + 1, y2 - y1 + 1
                aspect = box_width / box_height
                if 0.35 <= aspect <= 2.8:
                    status, confidence = self._classify_basic(image, (x1, y1, x2 + 1, y2 + 1))
                    detections.append(
                        Detection(
                            id=f"egg-{len(detections) + 1}",
                            label="Huevo",
                            status=status,
                            confidence=confidence,
                            box=[x1 / width, y1 / height, box_width / width, box_height / height],
                        )
                    )
        return detections[:500]


detector = EggDetector()


def detection_payload(session_id: str, sequence: int, detections: list[Detection]) -> dict[str, Any]:
    return {
        "type": "detections",
        "sessionId": session_id,
        "sequence": sequence,
        "detections": [
            {
                "id": detection.id,
                "label": detection.label,
                "status": detection.status,
                "confidence": detection.confidence,
                "box": detection.box,
            }
            for detection in detections
        ],
    }


def annotate_image(image: Image.Image, detections: list[Detection]) -> Image.Image:
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    width, height = output.size
    font_size = max(16, min(32, width // 18))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    for detection in detections:
        x, y, box_width, box_height = detection.box
        left, top = int(x * width), int(y * height)
        right, bottom = int((x + box_width) * width), int((y + box_height) * height)
        is_broken = detection.status == "Roto"
        color = (220, 35, 35) if is_broken else (20, 145, 70)
        draw.rectangle((left, top, right, bottom), outline=color, width=max(3, width // 180))
        text = f"HUEVO {detection.status.upper()} {detection.confidence:.0%}"
        text_left = right + 6
        text_top = top
        text_box = draw.textbbox((0, 0), text, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        if text_left + text_width + 6 > width:
            text_left = max(0, left - text_width - 6)
        if text_top + text_height + 6 > height:
            text_top = max(0, bottom - text_height - 6)
        background = (text_left, text_top, text_left + text_width + 8, text_top + text_height + 8)
        draw.rounded_rectangle(background, radius=4, fill=color)
        draw.text((text_left + 4, text_top + 3), text, fill=(255, 255, 255), font=font)
    return output


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/detect")
async def detect_image(file: UploadFile = File(...)) -> StreamingResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="El archivo debe ser una imagen")
    content = await file.read()
    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as error:
        raise HTTPException(status_code=400, detail="No se pudo leer la imagen") from error
    detections = detector.detect(image)
    annotated = annotate_image(image, detections)
    output = io.BytesIO()
    annotated.save(output, format="JPEG", quality=92)
    output.seek(0)
    return StreamingResponse(output, media_type="image/jpeg", headers={"X-Detection-Count": str(len(detections))})


def decode_frames(
    stream: IncrementalByteStream,
    frames: queue.Queue[Image.Image | None],
    width: int,
    height: int,
) -> None:
    process: subprocess.Popen[bytes] | None = None
    feeder: threading.Thread | None = None
    error_reader: threading.Thread | None = None

    def log_decoder_errors() -> None:
        assert process is not None and process.stderr is not None
        for line in process.stderr:
            logger.error("FFmpeg video: %s", line.decode("utf-8", errors="replace").strip())

    def feed_decoder() -> None:
        assert process is not None and process.stdin is not None
        try:
            while True:
                chunk = stream.read(64 * 1024)
                if not chunk:
                    break
                process.stdin.write(chunk)
                process.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        finally:
            try:
                process.stdin.close()
            except (BrokenPipeError, OSError):
                pass

    try:
        process = subprocess.Popen(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-loglevel",
                "error",
                "-flags",
                "low_delay",
                "-probesize",
                "1M",
                "-f",
                "webm",
                "-i",
                "pipe:0",
                "-f",
                "image2pipe",
                "-c:v",
                "bmp",
                "-pix_fmt",
                "bgr24",
                "pipe:1",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        error_reader = threading.Thread(target=log_decoder_errors, daemon=True)
        error_reader.start()
        feeder = threading.Thread(target=feed_decoder, daemon=True)
        feeder.start()
        assert process.stdout is not None
        first_frame = True
        while True:
            # BMP incluye longitud y dimensiones: no dividir bytes segun el start.
            header = process.stdout.read(14)
            if not header:
                break
            if len(header) != 14 or header[:2] != b"BM":
                raise ValueError("Cabecera de fotograma BMP incompleta o invalida")
            frame_size = int.from_bytes(header[2:6], "little")
            if not 14 < frame_size <= 128 * 1024 * 1024:
                raise ValueError("Tamano de fotograma invalido")
            pixels = process.stdout.read(frame_size - 14)
            if len(pixels) != frame_size - 14:
                raise ValueError("Fotograma de video incompleto")
            with Image.open(io.BytesIO(header + pixels)) as decoded:
                image = decoded.convert("RGB")
            if first_frame and image.size != (width, height):
                logger.warning("Dimensiones de video: frontend=%sx%s, reales=%sx%s; se usan las reales",
                               width, height, *image.size)
            first_frame = False
            try:
                frames.put_nowait(image)
            except queue.Full:
                try:
                    frames.get_nowait()
                except queue.Empty:
                    pass
                frames.put_nowait(image)
        feeder.join(timeout=1)
    except Exception:
        logger.exception("Error decodificando el flujo WebM")
    finally:
        stream.close()
        if process is not None:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            if feeder is not None:
                feeder.join(timeout=2)
            if error_reader is not None:
                error_reader.join(timeout=2)
            for pipe in (process.stdin, process.stdout, process.stderr):
                if pipe is not None:
                    pipe.close()
        try:
            frames.put_nowait(None)
        except queue.Full:
            pass


@app.websocket("/video")
async def video_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    stream = IncrementalByteStream()
    frames: queue.Queue[Image.Image | None] = queue.Queue(maxsize=1)
    decoder_task: asyncio.Task[None] | None = None
    inference_task: asyncio.Task[None] | None = None
    session_id: str | None = None
    closed = asyncio.Event()

    async def infer() -> None:
        sequence = 0
        next_inference = 0.0
        while not closed.is_set():
            await asyncio.sleep(max(0.0, next_inference - time.monotonic()))
            try:
                frame = await asyncio.to_thread(frames.get, True, 0.5)
            except queue.Empty:
                continue
            if frame is None:
                return
            if sequence == 0:
                logger.info("Video %s: primer fotograma decodificado (%s)", session_id, frame.size)
            try:
                started = time.monotonic()
                next_inference = started + 1.0 / VIDEO_MAX_FPS
                detections = await asyncio.to_thread(detector.detect, frame, imgsz=VIDEO_IMAGE_SIZE)
                elapsed_ms = (time.monotonic() - started) * 1000
            except Exception:
                logger.exception("Video %s: fallo de inferencia", session_id)
                await websocket.close(code=1011, reason="Error procesando el video")
                return
            sequence += 1
            if sequence == 1 or sequence % 30 == 0:
                logger.info("Video %s: prediccion %d, %d detecciones, %.1f ms, limite=%.1f FPS, imgsz=%d",
                            session_id, sequence, len(detections), elapsed_ms, VIDEO_MAX_FPS, VIDEO_IMAGE_SIZE)
            await websocket.send_text(json.dumps(detection_payload(session_id or "", sequence, detections), separators=(",", ":")))

    try:
        logger.info("Video: conexion abierta; esperando mensaje start")
        start = await websocket.receive_json()
        if start.get("type") != "start" or not isinstance(start.get("sessionId"), str) or not start["sessionId"]:
            await websocket.close(code=1008, reason="El primer mensaje debe ser start con sessionId")
            return
        session_id = start["sessionId"]
        mime_type = start.get("mimeType", "")
        if "webm" not in mime_type:
            await websocket.close(code=1003, reason="Solo se admite video WebM")
            return
        width = start.get("width")
        height = start.get("height")
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            await websocket.close(code=1008, reason="start requiere width y height validos")
            return
        logger.info("Video %s: start recibido, formato=%s, dimensiones=%sx%s", session_id, mime_type, width, height)
        received_chunks = 0
        decoder_task = asyncio.create_task(asyncio.to_thread(decode_frames, stream, frames, width, height))
        inference_task = asyncio.create_task(infer())
        while True:
            message = await websocket.receive()
            if message.get("bytes") is not None:
                received_chunks += 1
                if received_chunks == 1:
                    logger.info("Video %s: primer fragmento recibido, %d bytes", session_id, len(message["bytes"]))
                stream.write(message["bytes"])
            elif message.get("type") == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Error en la sesion WebSocket %s", session_id)
    finally:
        closed.set()
        stream.close()
        for task in (decoder_task, inference_task):
            if task is not None:
                task.cancel()
        await asyncio.gather(*(task for task in (decoder_task, inference_task) if task is not None), return_exceptions=True)
