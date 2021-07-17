"""
    Asynchronously Reading Video Frames
"""


from typing import Union
from threading import Lock, Thread
from cv2 import VideoCapture, imshow, waitKey, destroyAllWindows
