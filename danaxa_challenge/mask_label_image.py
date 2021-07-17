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
# Detecting labels and making masks
results = model.detect([rgb_image], verbose=0)
result = results[0]
# Extracting label names instead of indexes
detected_labels = [CLASS_NAMES[item] for item in result.get("class_ids")]
print(f"Labels: {detected_labels}")

# Visualizing masks
visualize.display_instances(image=rgb_image,
                            boxes=result['rois'],
                            masks=result['masks'],
                            class_ids=result['class_ids'],
                            class_names=CLASS_NAMES,
                            scores=result['scores'])
