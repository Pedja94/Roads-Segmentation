import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

import cv2
import os
import extractGraphUtils as utils

def createGraph(imagesPath, resultsPath, modelsPath, modelsBackbone, modeLR, ppParameters):

    models, preprocessInputFs = utils.loadModel(modelsPath, modelsBackbone, modeLR)

    for imgName in os.listdir(imagesPath):
        #load image
        img = cv2.imread(os.path.join(imagesPath, imgName)) 
        #run inference
        pred = utils.getPrediction(img, models, preprocessInputFs)
        #build skeleton
        ppPred, predSkeleton = utils.buildSkeleton(pred, ppParameters)
        #build graph
        G = utils.buildGraph(predSkeleton, ppParameters.min_graph_length_pix, img.shape[0])

        utils.showImages([img, pred, ppPred, predSkeleton], G, resultPath=os.path.join(resultsPath, imgName))

@hydra.main(config_path='config', config_name='extractGraph.yaml')
def main(cfg: DictConfig):

    model_paths = []
    for path in cfg.paths.models:
        model_paths.append(to_absolute_path(path))

    createGraph(
      to_absolute_path(cfg.paths.images),
      to_absolute_path(cfg.paths.results),
      model_paths,
      cfg.model.backbones,
      cfg.model.lr,
      cfg.pp_parameters
      )


if __name__ == "__main__":
    main()