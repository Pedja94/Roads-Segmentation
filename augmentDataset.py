import os
import cv2
import random

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

def randomCrop(image, mask, tileSize):

    x = random.randint(0, image.shape[1] - tileSize)
    y = random.randint(0, image.shape[0] - tileSize)

    img = image[y:y+tileSize, x:x+tileSize]
    mask = mask[y:y+tileSize, x:x+tileSize]

    return img, mask
    
def augImage(image, mask):

    pHFlip = random.uniform(0, 1)
    pVFlip = random.uniform(0, 1)
    pRot = random.uniform(0, 1)
    rotNumber = random.randint(1, 3)

    if (pHFlip < 0.25):
        cv2.flip(image, 1)
        cv2.flip(mask, 1)

    if (pVFlip < 0.25):
        cv2.flip(image, 0)
        cv2.flip(mask, 0)

    if (pRot < 0.5):
        if (rotNumber == 1):
            cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
        if (rotNumber == 2):
            cv2.rotate(image, cv2.ROTATE_180)
            cv2.rotate(mask, cv2.ROTATE_180)
        if (rotNumber == 3):
            cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image, mask


def augmentDataset(imgPath, labPath, images, augPath, maskValPixThd, numOfTiles, tileSize):

    augImgPath = os.path.join(augPath, 'images/')
    augMaskPath = os.path.join(augPath, 'labels/')

    os.mkdir(augImgPath)
    os.mkdir(augMaskPath)

    for imgName in images:
        print(imgName)
        image = cv2.imread(os.path.join(imgPath, imgName))
        mask = cv2.imread(os.path.join(labPath, imgName[:-1]))

        for it in range(numOfTiles):
            newImg, newMask = randomCrop(image, mask, tileSize)
            newImg, newMask = augImage(newImg, newMask)

            _ ,newMask = cv2.threshold(newMask, maskValPixThd, 255, cv2.THRESH_BINARY)

            newName = os.path.splitext(imgName)[0] + '_' + str(it) + '.png'

            cv2.imwrite(os.path.join(augImgPath, newName), newImg)
            cv2.imwrite(os.path.join(augMaskPath, newName), newMask)


def createDataset(imgPath, labPath, augDatasetPath, trainP, valP, testP, maskValPixThd, numOfTiles, tileSize):

    if (not (trainP + valP + testP == 100)):
        print('Invalid split for train validation and test sets.')
        return

    images = os.listdir(imgPath)
    random.shuffle(images)

    augTrainPath = os.path.join(augDatasetPath, 'train/')

    if (os.path.exists(augTrainPath)):
        os.remove(augTrainPath)

    os.mkdir(augTrainPath)

    trainNum = int((trainP / 100.) * len(images))

    print('Create train dataset in ' + augTrainPath)
    augmentDataset(imgPath, labPath, images[:trainNum], augTrainPath, maskValPixThd, numOfTiles, tileSize)

    augValPath = os.path.join(augDatasetPath, 'val/')

    if (os.path.exists(augValPath)):
        os.remove(augValPath)

    os.mkdir(augValPath)

    valNum = int((valP / 100.) * len(images))

    print('Create validation dataset in ' + augValPath)
    augmentDataset(imgPath, labPath, images[trainNum:trainNum+valNum], augValPath, maskValPixThd, numOfTiles, tileSize)

    augTestPath = os.path.join(augDatasetPath, 'test/')

    if (os.path.exists(augTestPath)):
        os.remove(augTestPath)

    os.mkdir(augTestPath)

    print('Create test dataset in ' + augTestPath)
    augmentDataset(imgPath, labPath, images[trainNum+valNum:], augTestPath, maskValPixThd, numOfTiles, tileSize)


@hydra.main(config_path='config', config_name='augment.yaml')
def main(cfg: DictConfig):

    createDataset(
      to_absolute_path(cfg.paths.images),
      to_absolute_path(cfg.paths.labels),
      to_absolute_path(cfg.paths.aug_path), 
      cfg.split.train,
      cfg.split.validation,
      cfg.split.test,
      cfg.params.pixel_thd,
      cfg.params.tile_num,
      cfg.params.tile_size
      )


if __name__ == "__main__":
    main()