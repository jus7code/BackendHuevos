"""Validated YOLO datasets and one background training process at a time."""
from __future__ import annotations

import asyncio
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import threading
import uuid
import zipfile

import yaml
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

BASE = Path(__file__).resolve().parent
MAX_UPLOAD = 512 * 1024 * 1024
MAX_EXPANDED = 2 * 1024 * 1024 * 1024
IMAGE_TYPES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
router = APIRouter(tags=['training'])


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False))
    temporary.replace(path)


def prepare_dataset(archive: Path, target: Path) -> dict:
    """Never execute or pass user YAML/download directives to Ultralytics."""
    target.mkdir()
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        if len(entries) > 20000 or sum(entry.file_size for entry in entries) > MAX_EXPANDED:
            raise ValueError('Dataset demasiado grande: maximo 20000 archivos y 2 GB extraidos')
        seen = set()
        for entry in entries:
            relative = PurePosixPath(entry.filename)
            if (relative.is_absolute() or '..' in relative.parts or '\\' in entry.filename
                    or (entry.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError('El ZIP contiene rutas no permitidas')
            if entry.is_dir():
                continue
            if relative in seen:
                raise ValueError('El ZIP contiene rutas duplicadas')
            seen.add(relative)
            if relative.suffix.lower() not in IMAGE_TYPES | {'.txt', '.yaml', '.yml'}:
                continue
            if entry.file_size > 32 * 1024 * 1024:
                raise ValueError('Un archivo excede 32 MB')
            destination = target.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(entry) as source, destination.open('wb') as output:
                shutil.copyfileobj(source, output)
    configs = list(target.rglob('data.yaml')) + list(target.rglob('data.yml'))
    if len(configs) != 1:
        raise ValueError('Incluye exactamente un data.yaml con names: [Sano, Roto]')
    config_path = configs[0]
    if config_path.stat().st_size > 65536:
        raise ValueError('data.yaml demasiado grande')
    config = yaml.safe_load(config_path.read_text())
    names = config.get('names') if isinstance(config, dict) else None
    if isinstance(names, dict):
        names = [names.get(0, names.get('0')), names.get(1, names.get('1'))] if len(names) == 2 else None
    if not isinstance(names, list) or [str(n).strip().lower() for n in names] != ['sano', 'roto']:
        raise ValueError('Las clases deben ser 0=Sano y 1=Roto, en ese orden')
    root = config_path.parent
    splits = {}
    counts = {}
    for split, aliases in (('train', ('train',)), ('val', ('val', 'valid'))):
        options = [(root / 'images' / alias, root / 'labels' / alias) for alias in aliases]
        options += [(root / alias / 'images', root / alias / 'labels') for alias in aliases]
        available = [(images, labels) for images, labels in options if images.is_dir() and labels.is_dir()]
        if len(available) != 1:
            raise ValueError(f'Faltan carpetas unicas de imagenes y etiquetas para {split}')
        images, labels = available[0]
        files = [p for p in images.rglob('*') if p.suffix.lower() in IMAGE_TYPES]
        if not files:
            raise ValueError(f'No hay imagenes en {split}')
        boxes = 0
        stems = set()
        for path in files:
            relative = path.relative_to(images).with_suffix('')
            if relative in stems:
                raise ValueError('Imagenes con el mismo nombre base en un split')
            stems.add(relative)
            with Image.open(path) as image:
                if image.width * image.height > 40_000_000:
                    raise ValueError('Una imagen excede 40 megapixeles')
                image.verify()
            label = labels / path.relative_to(images).with_suffix('.txt')
            if not label.is_file() or label.stat().st_size > 1024 * 1024:
                raise ValueError(f'Falta etiqueta TXT valida para {path.name}; usa TXT vacio para fondos')
            for line in label.read_text().splitlines():
                if not line.strip():
                    continue
                values = line.split()
                if len(values) != 5 or values[0] not in {'0', '1'}:
                    raise ValueError('Cada caja debe ser: clase x_centro y_centro ancho alto')
                x, y, width, height = map(float, values[1:])
                if (not all(math.isfinite(v) for v in (x, y, width, height))
                        or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1)
                        or x-width/2 < -0.001 or y-height/2 < -0.001
                        or x+width/2 > 1.001 or y+height/2 > 1.001):
                    raise ValueError('Las cajas deben estar normalizadas y dentro de la imagen')
                boxes += 1
        if not boxes:
            raise ValueError(f'El split {split} debe tener al menos una caja etiquetada')
        splits[split] = str(images.resolve())
        counts[split] = {'images': len(files), 'boxes': boxes}
    # Fixed configuration: ignore uploaded paths, scripts, URLs and download keys.
    safe_config = {'path': str(root.resolve()), **splits, 'names': ['Sano', 'Roto']}
    data_path = target / 'validated.yaml'
    data_path.write_text(yaml.safe_dump(safe_config))
    return {'data': str(data_path), 'counts': counts}


class TrainingManager:
    def __init__(self, root: Path):
        self.root = root
        self.lock = threading.Lock()
        self.active: tuple[Path, subprocess.Popen] | None = None

    def _refresh(self):
        if self.active is None:
            return
        directory, process = self.active
        code = process.poll()
        if code is None:
            return
        status = json.loads((directory / 'status.json').read_text())
        success = code == 0 and (directory / 'run/weights/best.pt').is_file()
        status.update(status='completed' if success else 'failed', exitCode=code)
        if not success:
            status['error'] = 'El entrenamiento fallo. Consulta el log del trabajo.'
        write_json(directory / 'status.json', status)
        self.active = None

    def start(self, directory: Path, epochs: int, imgsz: int, batch: int) -> dict:
        with self.lock:
            self._refresh()
            if self.active is not None:
                raise HTTPException(409, 'Ya hay un entrenamiento en curso')
            dataset = prepare_dataset(directory / 'dataset.zip', directory / 'dataset')
            config = {'data': dataset['data'], 'epochs': epochs, 'imgsz': imgsz, 'batch': batch}
            write_json(directory / 'config.json', config)
            status = {'jobId': directory.name, 'status': 'running', **config, 'dataset': dataset['counts']}
            status.pop('data')
            write_json(directory / 'status.json', status)
            env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
            with (directory / 'train.log').open('wb') as log:
                process = subprocess.Popen([sys.executable, '-m', 'app.train_worker', str(directory)],
                                           cwd=BASE.parent, env=env, stdout=log, stderr=subprocess.STDOUT)
            self.active = directory, process
            return status

    def get(self, job_id: str) -> tuple[Path, dict]:
        try:
            if str(uuid.UUID(job_id)) != job_id:
                raise ValueError()
        except ValueError:
            raise HTTPException(404, 'Trabajo no encontrado')
        with self.lock:
            self._refresh()
            directory = self.root / job_id
            path = directory / 'status.json'
            if not path.is_file():
                raise HTTPException(404, 'Trabajo no encontrado')
            status = json.loads(path.read_text())
            if status['status'] == 'running' and (self.active is None or self.active[0] != directory):
                status.update(status='interrupted', error='El servicio se reinicio durante el entrenamiento')
                write_json(path, status)
            progress = directory / 'progress.json'
            if progress.is_file():
                status['progress'] = json.loads(progress.read_text())
            if status['status'] == 'completed':
                status['modelUrl'] = f'/train/{job_id}/model'
            return directory, status

    def shutdown(self):
        with self.lock:
            if self.active is not None:
                _, process = self.active
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                self._refresh()


training_manager = TrainingManager(BASE.parent / 'training_jobs')


@router.post('/train', status_code=202)
async def train(file: UploadFile = File(...), epochs: int = Form(20, ge=1, le=100),
                imgsz: int = Form(416, ge=320, le=640), batch: int = Form(2, ge=1, le=8)):
    if imgsz % 32:
        raise HTTPException(422, 'imgsz debe ser multiplo de 32')
    directory = training_manager.root / str(uuid.uuid4())
    directory.mkdir(parents=True)
    try:
        size = 0
        with (directory / 'dataset.zip').open('wb') as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    raise HTTPException(413, 'El ZIP excede 512 MB')
                output.write(chunk)
        return await asyncio.to_thread(training_manager.start, directory, epochs, imgsz, batch)
    except HTTPException:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    except (ValueError, OSError, zipfile.BadZipFile, yaml.YAMLError) as error:
        shutil.rmtree(directory, ignore_errors=True)
        raise HTTPException(422, str(error)) from error
    finally:
        await file.close()


@router.get('/train/{job_id}')
def training_status(job_id: str):
    return training_manager.get(job_id)[1]


@router.get('/train/{job_id}/model')
def training_model(job_id: str):
    directory, status = training_manager.get(job_id)
    if status['status'] != 'completed':
        raise HTTPException(409, 'El modelo aun no esta disponible')
    return FileResponse(directory / 'run/weights/best.pt', filename=f'{job_id}-best.pt')


@router.get('/train/{job_id}/log')
def training_log(job_id: str):
    directory, _ = training_manager.get(job_id)
    return FileResponse(directory / 'train.log', media_type='text/plain')
