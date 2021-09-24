import segmentation_models as sm
from segmentation_models import get_preprocessing
from tensorflow.python.keras.models import load_model
import tensorflow as tf

import matplotlib.pyplot as plt

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

def showImages(img, pred):
    fig = plt.figure(figsize=(1, 2))

    fig.add_subplot(1, 2, 1)
    plt.imshow(img)
    fig.add_subplot(1, 2, 2)
    plt.imshow(pred)

    plt.show()

def postProcessPrediction(pred):
    a = 5