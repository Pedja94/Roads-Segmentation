import keras
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.python.keras.models import load_model
import tensorflow as tf

import argparse
import os

import cv2
import numpy as np
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

import models 

def load_data(imgPath, labPath):

  images = []
  masks = []

  for imgName in os.listdir(imgPath):
    images.append(cv2.imread(os.path.join(imgPath, imgName)))
    masks.append(cv2.imread(os.path.join(labPath, imgName)))

  return np.array(images), np.array(masks)


def generator(img_dir, label_dir, batch_size, input_size):
    list_images = os.listdir(img_dir)
    # shuffle(list_images) #Randomize the choice of batches
    ids_train_split = range(len(list_images))

    while True:
      for start in range(0, len(ids_train_split), batch_size):
        x_batch = []
        y_batch = []

        end = min(start + batch_size, len(ids_train_split))
        ids_train_batch = ids_train_split[start:end]

        for id in ids_train_batch:
          img_name = os.path.join(img_dir,str(list_images[id]))
          mask_name = os.path.join(label_dir, str(list_images[id]))

          img = cv2.imread(img_name) 
          mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)               
          
          x_batch += [img]
          y_batch += [mask]    


        x_batch = np.array(x_batch) / 255.
        y_batch = np.array(y_batch) / 255.

        yield x_batch, np.expand_dims(y_batch, -1)


def createAndTrainModel(trainImgPath, trainLabPath, testImgPath, testLabPath, valImgPath, valLabPath,
                        imageSize, learningRate, batchSize, epochs, modelName, logFile):

    model = models.unet_16_256([imageSize, imageSize, 3])
    #model.summary()

    opt = tf.keras.optimizers.Adam(learning_rate=learningRate)

    model.compile(
      optimizer=opt,
      loss=models.soft_dice_loss,
      metrics=[models.iou_coef])

    logger = CSVLogger(logFile,
      separator=',',
      append=True
    )

    checkpointer = ModelCheckpoint(modelName,
                             monitor="val_loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)

    # earlystopper = EarlyStopping(monitor = 'val_loss', 
    #                       min_delta = 0, 
    #                       patience = 5,
    #                       verbose = 1,
    #                       restore_best_weights = True)

    lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=4,
                               verbose=1,
                               epsilon=1e-4)
                         
    history = model.fit(generator(trainImgPath, trainLabPath, batchSize, [imageSize, imageSize, 3]),                              
                              steps_per_epoch=len(os.listdir(trainImgPath))/batchSize,
                              epochs=epochs,
                              verbose=1,
                              callbacks= [checkpointer, lr_reducer, logger],
                              validation_data=generator(valImgPath, valLabPath, batchSize, [imageSize, imageSize, 3]),
                              validation_steps=1,
                              class_weight=None,
                              max_queue_size=10,
                              workers=1
                              )

    model = load_model(modelName, custom_objects={'soft_dice_loss': models.soft_dice_loss, 'iou_coef': models.iou_coef})
    model.compile(
      optimizer=opt,
      loss=models.soft_dice_loss,
      metrics=[models.iou_coef])
    
    test_gen = generator(testImgPath, testLabPath, 1, [imageSize, imageSize, 3])
    model.evaluate(
      test_gen,
      #steps =len(os.listdir(testImgPath))/batchSize,
      steps=1
    )


@hydra.main(config_path='config', config_name='train.yaml')
def main(cfg: DictConfig):

    createAndTrainModel(
      to_absolute_path(cfg.paths.val.images),
      to_absolute_path(cfg.paths.val.labels),
      to_absolute_path(cfg.paths.test.images),
      to_absolute_path(cfg.paths.test.labels),
      to_absolute_path(cfg.paths.val.images),
      to_absolute_path(cfg.paths.val.labels), 
      cfg.training.imageSize[0],
      cfg.training.lr,
      cfg.training.batchSize,
      cfg.training.epochs,
      cfg.output.model,
      cfg.output.log
      )


if __name__ == "__main__":
    main()