import argparse
import os
import cv2
import numpy as np
import random

def invalidImage(image, whitePixThd):

    nTotalPix = image.shape[0] * image.shape[1]
    nWhitePix = np.sum(image[:, :] == [255, 255, 255])

    return (nWhitePix / nTotalPix) > whitePixThd


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


def augmentDataset(imgPath, labPath, augPath, whitePixThd, numOfTiles, tileSize):

    augImgPath = os.path.join(augPath, 'images/')
    augMaskPath = os.path.join(augPath, 'labels/')

    os.mkdir(augImgPath)
    os.mkdir(augMaskPath)

    for imgName in os.listdir(imgPath):
        print(imgName)
        image = cv2.imread(os.path.join(imgPath, imgName))
        mask = cv2.imread(os.path.join(labPath, imgName[:-1]))

        if (invalidImage(image, whitePixThd)):
            print ('Too many white pixels')
            continue

        for it in range(numOfTiles):
            newImg, newMask = randomCrop(image, mask, tileSize)
            newImg, newMask = augImage(newImg, newMask)

            newName = os.path.splitext(imgName)[0] + '_' + str(it) + '.png'

            cv2.imwrite(os.path.join(augImgPath, newName), newImg)
            cv2.imwrite(os.path.join(augMaskPath, newName), newMask)


def createDataset(trainImgPath, trainLabPath, testImgPath, testLabPath, valImgPath, valLabPath,
                    augDatasetPath, whitePixThd, numOfTiles, tileSize):

    augTrainPath = os.path.join(augDatasetPath, 'train/')

    if (os.path.exists(augTrainPath)):
        os.remove(augTrainPath)

    os.mkdir(augTrainPath)

    print('Create train dataset in ' + augTrainPath)
    augmentDataset(trainImgPath, trainLabPath, augTrainPath, whitePixThd, numOfTiles, tileSize)

    augValPath = os.path.join(augDatasetPath, 'val/')

    if (os.path.exists(augValPath)):
        os.remove(augValPath)

    os.mkdir(augValPath)

    print('Create validation dataset in ' + augValPath)
    augmentDataset(valImgPath, valLabPath, augValPath, whitePixThd, numOfTiles, tileSize)

    augTestPath = os.path.join(augDatasetPath, 'test/')

    if (os.path.exists(augTestPath)):
        os.remove(augTestPath)

    os.mkdir(augTestPath)

    print('Create test dataset in ' + augTestPath)
    augmentDataset(testImgPath, testLabPath, augTestPath, whitePixThd, numOfTiles, tileSize)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--trainImages', type=str, default='./Dataset/tiff/train/', help="Images from train set.")
    parser.add_argument('--trainLabels', type=str, default='./Dataset/tiff/train_labels/', help="Labels from train set.")
    parser.add_argument('--valImages', type=str, default='./Dataset/tiff/val/', help="Images from validation set.")
    parser.add_argument('--valLabels', type=str, default='./Dataset/tiff/val_labels/', help="Labels from validation set.")
    parser.add_argument('--testImages', type=str, default='./Dataset/tiff/test/', help="Images from test set.")
    parser.add_argument('--testLabels', type=str, default='./Dataset/tiff/test_labels/', help="Labels from test set.")
    parser.add_argument('--augDataset', type=str, default='./AugmentedDataset/', help="Labels from validation set.")
    parser.add_argument('--whitePixThd', type=float, default=0.50, help="Precentage of white pixels on image.")
    parser.add_argument('--numOfTiles', type=int, default=25, help="Number of tiles extracted from one image.")
    parser.add_argument('--tileSize', type=int, default=256, help="Tile size(both height and width).")

    args = parser.parse_args()

    createDataset(args.trainImages, args.trainLabels, args.testImages, args.testLabels, args.valImages, args.valLabels,
                    args.augDataset, args.whitePixThd, args.numOfTiles, args.tileSize)


if __name__ == "__main__":
    main()