
import tensorflow as tf
import numpy as np
import streamlit as st
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import cv2
from mrcnn.model import mold_image
#from mrcnn.visualize import display_instances
from mrcnn.visualize import apply_mask
from mrcnn.visualize import random_colors

from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import IPython.display
from keras import backend as K
from keras.backend import clear_session


class PredictionConfig(Config):
    NAME = "pred_cfg"


    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1 + 1 + 1

    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024

    RPN_ANCHOR_SCALES = (8, 16, 64, 128, 256)  # anchor side in pixels

    TRAIN_ROIS_PER_IMAGE = 600

    STEPS_PER_EPOCH = 150

    USE_MINI_MASK = False

    DETECTION_MAX_INSTANCES = 5
    
    #DETECTION_MIN_CONFIDENCE = 0.85

    DETECTION_MIN_CONFIDENCE = 0.7

    DETECTION_NMS_THRESHOLD = 0.3

cfg = PredictionConfig()

    
    
    



st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Breast Cancer Detection - BAI")



uploaded_file = st.file_uploader("Choose a image file", type="png")



if uploaded_file is not None:
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    
    
    
    #@st.cache(allow_output_mutation=True)
    def load_model():
      model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
      model.load_weights(r'C:\Users\DELL PC\Desktop\MaskRCNN\streamlit\weights.h5', by_name=True)
      #model.keras_model._make_predict_function()
    
      #weight64 is best
      return model
    
    with st.spinner('Loading Model Into Memory....'):
       model = load_model()
    
        
    #image = cv2.imread('\image1.png')
    def detect(image , config):
        #global model
    
        scaled_image = mold_image(image, config)
        sample = np.expand_dims(scaled_image, 0)
        
        return model.detect(sample, verbose=0) , scaled_image
    
    
    
    yhat, scaled_image = detect(image, cfg)
    
    
    def display_instances(image, boxes, masks, class_ids, class_names,
                          scores=None, title="",
                          figsize=(16, 16), ax=None,
                          show_mask=True, show_bbox=True,
                          colors=None, captions=None):
        """
        boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
        masks: [height, width, num_instances]
        class_ids: [num_instances]
        class_names: list of class names of the dataset
        scores: (optional) confidence scores for each box
        title: (optional) Figure title
        show_mask, show_bbox: To show masks and bounding boxes or not
        figsize: (optional) the size of the image
        colors: (optional) An array or colors to use with each object
        captions: (optional) A list of strings to use as captions for each object
        """
        N = boxes.shape[0]
        if not N:
            print("\n*** No instances to display *** \n")
        else:
            assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
    
        auto_show = False
        if not ax:
            _, ax = plt.subplots(1, figsize=figsize)
            auto_show = True
    
        colors = colors or random_colors(N)
    
        height, width = image.shape[:2]
        ax.set_ylim(height + 10, -10)
        ax.set_xlim(-10, width + 10)
        ax.axis('off')
        ax.set_title(title)
    
        masked_image = image.astype(np.uint32).copy()
        for i in range(N):
            color = colors[i]
    
            if not np.any(boxes[i]):
                continue
            y1, x1, y2, x2 = boxes[i]
            if show_bbox:
                p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                    alpha=0.7, linestyle="dashed",
                                    edgecolor=color, facecolor='none')
                ax.add_patch(p)
    
            if not captions:
                class_id = class_ids[i]
                score = scores[i] if scores is not None else None
                label = class_names[class_id]
                caption = "{} {:.3f}".format(label, score) if score else label
            else:
                caption = captions[i]
            ax.text(x1, y1 + 8, caption,
                    color='w', size=11, backgroundcolor="none")
    
            mask = masks[:, :, i]
            if show_mask:
                masked_image = apply_mask(masked_image, mask, color)
    
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
        ax.imshow(masked_image.astype(np.uint8))
        st.pyplot()
    
    
    display_instances(scaled_image, yhat[0]['rois'], yhat[0]['masks'], yhat[0]['class_ids'], ['Benign' , 'Malignant' , 'suspicious_anomaly'])
