"""
    Main file that contains following items:
        - Configuring and loading Mask-RCNN Model
        - Testing Labeling functionality
"""


from mrcnn import model, config
from cv2 import VideoCapture, cvtColor, COLOR_BGR2RGB


class MainConfig(config.Config):
    """ Model Configuration Class """
    NAME = "coco_inference"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 81


# Instantiating model and passing its configuration parameters
model = model.MaskRCNN(
    mode="inference",
    config=MainConfig(),
    model_dir="../models/logs"
)
