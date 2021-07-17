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


def stream(file_path: str = "videos/test_video.mkv"):
    """
    Displaying The Video File
    :param file_path: video file path(directory address)
    :return: None
    """
    video = VideoCaptureAsync(src=file_path)
    while True:
        retrieved, frame = video.capture.read()
        if retrieved:
            imshow("Test Video", frame)
            waitKey(10)
        if waitKey(1) & 0xFF == ord("q"):
            video.stop()
            destroyAllWindows()
            break
    video.stop()
    destroyAllWindows()


if __name__ == '__main__':
    # Streaming test video
    stream()
