# danaxa_challenge
Danaxa Technical Interview Challenge


## Challenge Description
Suppose we are developing a new labeling tool to annotate masks in a video.
Labeling all the frames of a video with a great accuracy takes a lot of time and
cost. In order to make annotation process faster, we need to use semi-
automated or automated labelling methods. Therefore, we need to implement
a method to annotate an object in a few frames and the tool keep detecting
that object in next frames. Ultimately, we want the tool to annotate all video
frames itself after annotating a few frames.

## Answering Questions
* Review previous work flow on this research topic and report it to us:
    * One shot learning 
    * Few shot learning
    * Siamese neural network
    * video object segmentation  

* Find recent works and publications which address this problem:
    * [Boundary-preserving Mask R-CNN](https://arxiv.org/abs/2007.08921)
    * [CANet: Class-Agnostic Segmentation Networks with Iterative Refinement and Attentive Few-Shot Learning](https://arxiv.org/abs/1903.02351)
    * [Learning What to Learn for Video Object Segmentation](https://arxiv.org/abs/2003.11540)
    * [Meta-DETR: Few-Shot Object Detection via Unified Image-Level Meta-Learning](https://arxiv.org/abs/2103.11731)
    * [Video Object Segmentation with Adaptive Feature Bank and Uncertain-Region Refinement](https://arxiv.org/abs/2010.07958)
    * [One-Shot Object Detection with Co-Attention and Co-Excitation](https://arxiv.org/abs/1911.12529)
    * [One-Shot Video Object Segmentation](https://arxiv.org/abs/1611.05198)
    * [Show&Tell: A Semi-Automated Image Annotation System](https://www.researchgate.net/publication/3338590_ShowTell_A_Semi-Automated_Image_Annotation_System)
    
* Find open source repositories in this regard:
    * [cvat](https://github.com/openvinotoolkit/cvat)
    * [Siamese-Networks-for-One-Shot-Learning](https://github.com/tensorfreitas/Siamese-Networks-for-One-Shot-Learning)
    * [keras-oneshot](https://github.com/sorenbouma/keras-oneshot)
    * [DPGN](https://github.com/megvii-research/DPGN)
    * [SSTVOS](https://github.com/dukebw/SSTVOS)
    * [pytracking](https://github.com/visionml/pytracking)
    
## Project Structure and Contents
images: contains a test image  
videos: contains a test video  
mrcnn: contains mask-rcnn model files  
weights: contains .h5 file (mask-rcnn model weights)  
arvf.py: contains a class in order to read video frames asynchronously  
main.py: contains loading, configuring and testing mask-rcnn model on test video  
mask_label_image.py: contains testing mask and label functionality on test image  
