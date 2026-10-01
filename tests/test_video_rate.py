import queue
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


class VideoRateTests(unittest.TestCase):
    def test_limits_inference_and_keeps_recent_frame(self):
        calls=[]

        def decode(stream, frames, width, height):
            for index in range(35):
                if stream._closing: break
                frame=Image.new('RGB',(32,32),(index,0,0))
                try: frames.put_nowait(frame)
                except queue.Full:
                    try: frames.get_nowait()
                    except queue.Empty: pass
                    frames.put_nowait(frame)
                time.sleep(1/30)

        def detect(frame, *, imgsz):
            calls.append((time.monotonic(),frame.getpixel((0,0))[0],imgsz))
            return []

        with patch('app.main.decode_frames',decode), patch('app.main.detector.detect',detect):
            with TestClient(app) as client:
                with client.websocket_connect('/video') as ws:
                    ws.send_json({'type':'start','sessionId':'rate-test','mimeType':'video/webm','width':32,'height':32})
                    results=[ws.receive_json() for _ in range(4)]
        self.assertEqual([r['sequence'] for r in results],[1,2,3,4])
        self.assertTrue(all(r['sessionId']=='rate-test' for r in results))
        self.assertTrue(all(c[2]==416 for c in calls))
        for previous,current in zip(calls,calls[1:]):
            self.assertGreaterEqual(current[0]-previous[0],0.18)
            self.assertGreater(current[1]-previous[1],1)
