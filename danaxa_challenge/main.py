"""
    Main file that contains following items:
        - Configuring and loading Mask-RCNN Model
        - Testing Labeling functionality
"""


from mrcnn import model, config
from cv2 import VideoCapture, cvtColor, COLOR_BGR2RGB
