"""
    Testing mask and label functionality on test image
"""


from mrcnn import visualize
from cv2 import imread, cvtColor, COLOR_BGR2RGB
from danaxa_challenge.main import model, CLASS_NAMES


file_path = "images/test_image.jpeg"
# Reading image and converting int channels
bgr_image = imread(file_path)
rgb_image = cvtColor(bgr_image, COLOR_BGR2RGB)
