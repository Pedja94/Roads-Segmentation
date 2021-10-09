from numba.core.types.misc import Undefined
import segmentation_models as sm
from segmentation_models import get_preprocessing
from tensorflow.python.keras.models import load_model
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.morphology import skeletonize

from graphUtils import sknw
from graphUtils.utils import cleanUpSmallEdges

def loadModel(modelsPath, modelsBackbone, modeLR): 
    sm.set_framework('tf.keras')
    models = []
    preprocessInputFs = []

    for i in range(len(modelsPath)):

        model = load_model(
            modelsPath[i], 
            custom_objects={'dice_loss': sm.losses.dice_loss, 'iou_score': sm.metrics.iou_score})

        opt = tf.keras.optimizers.Adam(learning_rate=modeLR)

        model.compile(
            optimizer=opt,
            loss=sm.losses.dice_loss,
            metrics=[sm.metrics.IOUScore(threshold=0.5)])
        
        preprocessInput = get_preprocessing(modelsBackbone[i])

        models.append(model)
        preprocessInputFs.append(preprocessInput)

    return models, preprocessInputFs

def getPrediction(img, models, preprocessInputFs):
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predArray = []

    for i in range(len(models)):
        imgTmp = np.copy(img)
        imgTmp = (imgTmp / 255.) if preprocessInputFs[i] is None else preprocessInputFs[i](imgTmp)

        #run inference
        predTmp = models[i](np.expand_dims(imgTmp, axis=0))
        predArray.append(np.squeeze(predTmp, axis=0))

    pred = np.mean(predArray, axis=0)

    return pred

def showImages(imgList, graph = Undefined, graphOverImg = True, resultPath = './result.png'):
    fig = plt.figure(figsize=(16, 14))
    numOfImages = len(imgList)

    for i in range(numOfImages):
        fig.add_subplot((numOfImages+1)/2 + 1, (numOfImages+1)/2 + 1, i+1)
        plt.imshow(imgList[i])

    #draw graph
    if (graph != Undefined):
        ax = fig.add_subplot((numOfImages+1)/2 + 1, (numOfImages+1)/2 + 1, numOfImages+1)

        if (graphOverImg and len(imgList) > 0):
            plt.imshow(imgList[0])
        # draw edges by pts
        for (s,e) in graph.edges():
            ps = graph[s][e]['pts']
            plt.plot(ps[:,1], ps[:,0], 'red')
        
        # draw node by o
        nodes = graph.nodes()
        ps = np.array([nodes[i]['o'] for i in nodes])
        if len(ps) != 0:
            plt.plot(ps[:,1], ps[:,0], 'b.')

        #save result
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(resultPath, bbox_inches=extent)

    plt.show()


def postProcess(pred, ppParameters):

    kernel_close = np.ones((ppParameters.kernel_close, ppParameters.kernel_close), np.uint8)
    kernel_open = np.ones((ppParameters.kernel_open, ppParameters.kernel_open), np.uint8)
    kernel_blur = ppParameters.kernel_close

    # global thresh      
    blur = cv2.medianBlur(pred, kernel_blur)
    glob_thresh_arr = cv2.threshold(blur, ppParameters.threshold, 1, cv2.THRESH_BINARY)[1]
    glob_thresh_arr_smooth = cv2.medianBlur(glob_thresh_arr, kernel_blur)
    mask_thresh = glob_thresh_arr_smooth      

    # opening and closing
    closing_t = cv2.morphologyEx(mask_thresh, cv2.MORPH_CLOSE, kernel_close)
    opening_t = cv2.morphologyEx(closing_t, cv2.MORPH_OPEN, kernel_open)
    blur = cv2.medianBlur(opening_t, kernel_blur)
    closing_t = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel_close)
    opening_t = cv2.morphologyEx(closing_t, cv2.MORPH_OPEN, kernel_open)

    img = opening_t.astype(np.bool)

    return img

def buildSkeleton(pred, ppParameters):
    rec = ppParameters.replicate_pix + ppParameters.clip_pix

    pred = cv2.copyMakeBorder(pred, ppParameters.replicate_pix, ppParameters.replicate_pix, 
                                ppParameters.replicate_pix, ppParameters.replicate_pix, 
                                cv2.BORDER_REPLICATE)  

    ppPred = postProcess(pred, ppParameters)

    # convert to 255 ramge
    pred = pred * 255
    pred = pred.astype(np.uint8)
    ske = skeletonize(ppPred).astype(np.uint16)

    ske = ske[rec:-rec, rec:-rec]
    ske = cv2.copyMakeBorder(ske, ppParameters.clip_pix, ppParameters.clip_pix,
                             ppParameters.clip_pix, ppParameters.clip_pix, 
                             cv2.BORDER_CONSTANT, value=0)
    ppPred = ppPred[ ppParameters.replicate_pix: -ppParameters.replicate_pix, 
                    ppParameters.replicate_pix: -ppParameters.replicate_pix ]

    return ppPred, ske

def buildGraph(skeleton, min_graph_length_pix, pix_extent):

    G = sknw.build_sknw(skeleton, multi=False)
    G = cleanUpSmallEdges(G, min_graph_length_pix, pix_extent)

    return G