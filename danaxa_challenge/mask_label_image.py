"""
    Testing mask and label functionality on test image
"""


from mrcnn import visualize
from cv2 import imread, cvtColor, COLOR_BGR2RGB
from danaxa_challenge.main import model, CLASS_NAMES
