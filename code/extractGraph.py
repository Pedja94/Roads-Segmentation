import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

import os
import cv2
import numpy as np
from PIL import Image 
import PIL 
import matplotlib.pyplot as plt

import tensorflow as tf

import extractGraphUtils as utils

def createGraph(imagesPath, resultsPath, modelPath, modelBackbone, modeLR):

    model, preprocessInput = utils.loadModel(modelPath, modelBackbone, modeLR)

    for imgName in os.listdir(imagesPath):
        #load and prepare image
        img = cv2.imread(os.path.join(imagesPath, imgName)) 
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img / 255.) if preprocessInput is None else preprocessInput(img)

        #run inference
        pred = model(np.expand_dims(img, axis=0))
        pred = np.squeeze(pred, axis=0)

        #build skeleton
        ppPred, predSkeleton = utils.buildSkeleton(pred)

        G = utils.buildGraph(predSkeleton)

        utils.showImages([img, pred, ppPred, predSkeleton])

@hydra.main(config_path='config', config_name='extractGraph.yaml')
def main(cfg: DictConfig):

    createGraph(
      to_absolute_path(cfg.paths.images),
      to_absolute_path(cfg.paths.results),
      to_absolute_path(cfg.paths.model),
      cfg.model.backbone,
      cfg.model.lr
      )


if __name__ == "__main__":
    main()