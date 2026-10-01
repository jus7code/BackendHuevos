"""Isolated fine-tuning; never replaces the model serving predictions."""
import json
import os
from pathlib import Path
import sys

import torch
from ultralytics import YOLO

from app.training import BASE, write_json


def run(directory: Path):
    os.nice(10)
    torch.set_num_threads(1)
    config = json.loads((directory / 'config.json').read_text())
    model = YOLO(str(BASE / 'modelo_nuevo' / 'best (1).pt'))

    def progress(trainer):
        write_json(directory / 'progress.json', {'epoch': min(trainer.epoch + 1, config['epochs']), 'epochs': config['epochs']})

    model.add_callback('on_fit_epoch_end', progress)
    model.train(**config, project=str(directory), name='run', exist_ok=False,
                device='cpu', workers=0, amp=False, cache=False, plots=False,
                save=True, val=True, close_mosaic=0)
    if not (directory / 'run/weights/best.pt').is_file():
        raise RuntimeError('El entrenamiento no genero best.pt')


if __name__ == '__main__':
    run(Path(sys.argv[1]).resolve())
