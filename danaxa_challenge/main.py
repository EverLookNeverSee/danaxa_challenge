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

# Loading model weights
weights = "../models/weights/mask_rcnn_coco.h5"
model.load_weights(filepath=weights, by_name=True)

# Model class names
CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
               'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
               'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
               'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
               'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
               'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
               'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
