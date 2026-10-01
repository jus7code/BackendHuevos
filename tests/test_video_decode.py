import io
import queue
import subprocess
import threading
import unittest

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

from app.main import IncrementalByteStream, decode_frames


class VideoDecodeTests(unittest.TestCase):
    def test_actual_dimensions_and_rgb_survive_fragmented_webm(self):
        original = Image.new('RGB', (160, 96), 'red')
        draw = ImageDraw.Draw(original)
        draw.rectangle((80, 0, 159, 47), fill='blue')
        draw.rectangle((0, 48, 79, 95), fill='lime')
        draw.rectangle((80, 48, 159, 95), fill='white')
        png = io.BytesIO()
        original.save(png, format='PNG')
        for codec in ('libvpx', 'libvpx-vp9'):
            with self.subTest(codec=codec):
                encoded = subprocess.run([
                    imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error',
                    '-f', 'image2pipe', '-i', 'pipe:0', '-frames:v', '1',
                    '-c:v', codec, '-f', 'webm', 'pipe:1',
                ], input=png.getvalue(), capture_output=True, check=True).stdout
                stream = IncrementalByteStream()
                frames = queue.Queue(maxsize=10)
                worker = threading.Thread(target=decode_frames,
                                          args=(stream, frames, 128, 128), daemon=True)
                worker.start()
                for offset in range(0, len(encoded), 73):
                    stream.write(encoded[offset:offset + 73])
                stream.close()
                worker.join(timeout=10)
                self.assertFalse(worker.is_alive(), 'Decoder failed to terminate')
                frame = frames.get(timeout=2)
                self.assertIsNotNone(frame)
                self.assertEqual(frame.size, original.size)
                self.assertEqual(frame.mode, 'RGB')
                # Interior pixels avoid normal chroma subsampling at boundaries.
                for x, y in ((30, 20), (120, 20), (30, 70), (120, 70)):
                    delta = np.abs(np.array(frame.getpixel((x, y)), dtype=int)
                                   - np.array(original.getpixel((x, y)), dtype=int))
                    self.assertLessEqual(int(delta.max()), 15)


if __name__ == '__main__':
    unittest.main()
