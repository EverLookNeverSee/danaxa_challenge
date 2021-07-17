"""
    Asynchronously Reading Video Frames
"""


from typing import Union
from threading import Lock, Thread
from cv2 import VideoCapture, imshow, waitKey, destroyAllWindows


class VideoCaptureAsync(object):
    """ Capturing Video frames Asynchronously """
    def __init__(self, src: Union[int, str] = 0):
        self.source = src
        self.capture = VideoCapture(self.source)
        self.grabbed, self.frame = self.capture.read()
        self.started = False
        self.thread = None
        self.read_lock = Lock()

    def set(self, key, value):
        self.capture.set(key, value)

    def start(self):
        if self.started:
            print("[Warning] Asynchronous video capturing is already started.")
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.capture.release()
