import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import zipfile

from fastapi import HTTPException
from PIL import Image

from app.training import prepare_dataset, TrainingManager


def dataset_zip(path, *, names='[Sano, Roto]', label='0 0.5 0.5 0.4 0.4', extra=None):
    image=io.BytesIO(); Image.new('RGB',(32,32),'white').save(image,format='PNG')
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('data.yaml',f'names: {names}\ndownload: echo NEVER_EXECUTE\n')
        for split in ('train','val'):
            z.writestr(f'images/{split}/egg.png',image.getvalue())
            z.writestr(f'labels/{split}/egg.txt',label)
        if extra: z.writestr(*extra)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.archive=self.root/'dataset.zip'

    def test_valid_dataset_uses_generated_safe_yaml(self):
        dataset_zip(self.archive)
        result=prepare_dataset(self.archive,self.root/'data')
        self.assertEqual(result['counts']['train'],{'images':1,'boxes':1})
        self.assertNotIn('download',Path(result['data']).read_text())

    def test_rejects_path_traversal(self):
        dataset_zip(self.archive,extra=('../escaped.txt','no'))
        with self.assertRaises(ValueError): prepare_dataset(self.archive,self.root/'data')
        self.assertFalse((self.root/'escaped.txt').exists())

    def test_rejects_reversed_classes(self):
        dataset_zip(self.archive,names='[Roto, Sano]')
        with self.assertRaises(ValueError): prepare_dataset(self.archive,self.root/'data')

    def test_rejects_invalid_coordinates(self):
        for index,label in enumerate(('0 nan 0.5 0.2 0.2','0 0.9 0.5 0.8 0.2','2 0.5 0.5 0.2 0.2')):
            with self.subTest(label=label):
                dataset_zip(self.archive,label=label)
                with self.assertRaises(ValueError): prepare_dataset(self.archive,self.root/f'data{index}')

    def test_only_one_training_job(self):
        first=self.root/'first'; first.mkdir(); dataset_zip(first/'dataset.zip')
        manager=TrainingManager(self.root)
        process=MagicMock(); process.poll.return_value=None
        with patch('app.training.subprocess.Popen',return_value=process):
            manager.start(first,1,320,2)
            with self.assertRaises(HTTPException) as caught:
                manager.start(self.root/'second',1,320,2)
            self.assertEqual(caught.exception.status_code,409)
        process.poll.return_value=1
        manager._refresh()
        self.assertIsNone(manager.active)
