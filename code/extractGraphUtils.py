import segmentation_models as sm
from segmentation_models import get_preprocessing
from tensorflow.python.keras.models import load_model
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.morphology import skeletonize

def loadModel(weightsPath, modelBackbone, modeLR): 
    sm.set_framework('tf.keras')
    model = load_model(
        weightsPath, 
        custom_objects={'dice_loss': sm.losses.dice_loss, 'iou_score': sm.metrics.iou_score})

    opt = tf.keras.optimizers.Adam(learning_rate=modeLR)

    model.compile(
        optimizer=opt,
        loss=sm.losses.dice_loss,
        metrics=[sm.metrics.IOUScore(threshold=0.5)])
    
    preprocessInput = get_preprocessing(modelBackbone)

    return model, preprocessInput

def showImages(imgList):
    fig = plt.figure(figsize=(15, 4))
    numOfImages = len(imgList)

    for i in range(numOfImages):
        fig.add_subplot(1, numOfImages, i+1)
        plt.imshow(imgList[i])

    plt.show()

def postProcess(pred):
    #params
    kernel_close_size = 3
    kernel_open_size = 3
    thresh = 0.3

    kernel_close = np.ones((kernel_close_size, kernel_close_size), np.uint8)
    kernel_open = np.ones((kernel_open_size, kernel_open_size), np.uint8)
    kernel_blur = kernel_close_size

    # global thresh
    #mask_thresh = (img > (img_mult * thresh))#.astype(np.bool)        
    blur = cv2.medianBlur(pred, kernel_blur)
    glob_thresh_arr = cv2.threshold(blur, thresh, 1, cv2.THRESH_BINARY)[1]
    glob_thresh_arr_smooth = cv2.medianBlur(glob_thresh_arr, kernel_blur)
    mask_thresh = glob_thresh_arr_smooth      

    # opening and closing
    # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    #gradient = cv2.morphologyEx(mask_thresh, cv2.MORPH_GRADIENT, kernel)
    closing_t = cv2.morphologyEx(mask_thresh, cv2.MORPH_CLOSE, kernel_close)
    opening_t = cv2.morphologyEx(closing_t, cv2.MORPH_OPEN, kernel_open)
    img = opening_t.astype(np.bool)

    return img

def buildSkeleton(pred):
    #params
    replicate = 3 #replicate border
    clip = 1
    rec = replicate + clip

    # convert to 255 ramge
    pred = pred * 255
    pred = pred.astype(np.uint8)

    pred = cv2.copyMakeBorder(pred, replicate, replicate, replicate, 
                                 replicate, cv2.BORDER_REPLICATE)  

    ppPred = postProcess(pred)
    ske = skeletonize(ppPred).astype(np.uint16)

    ske = ske[rec:-rec, rec:-rec]
    ske = cv2.copyMakeBorder(ske, clip, clip, clip, clip, cv2.BORDER_CONSTANT, value=0)
    ppPred = ppPred[replicate:-replicate,replicate:-replicate]

    return ppPred, ske

def buildGraph(skeleton):
    return skeleton