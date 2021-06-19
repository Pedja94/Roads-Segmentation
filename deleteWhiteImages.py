import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import os

def deleteImages(imagesPath, masksPath, toDeleteFilePath):

    file = open(toDeleteFilePath, 'r')
    txtFileLines = file.readlines()
 
    for line in txtFileLines:
        name = line.strip()
        imgName = os.path.join(imagesPath, name + '_15.tiff')
        maskName = os.path.join(masksPath, name + '_15.tif')

        if os.path.exists(imgName):
            os.remove(imgName)

        if os.path.exists(maskName):
            os.remove(maskName)


@hydra.main(config_path='config', config_name='deleteWhiteImages.yaml')
def main(cfg: DictConfig):

    deleteImages(
      to_absolute_path(cfg.paths.dataset.images),
      to_absolute_path(cfg.paths.dataset.labels),
      to_absolute_path(cfg.paths.delete_file)
      )


if __name__ == "__main__":
    main()