import argparse
import models 
import keras
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.python.keras.models import load_model
import tensorflow as tf
import os
import cv2
import numpy as np

def train_generator(img_dir, label_dir, batch_size, input_size):
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
              img_name = img_dir + list_images[id]
              mask_name = label_dir + list_images[id]
  
              img = cv2.imread(img_name) 
              mask = cv2.imread(mask_name)               
              
              x_batch += [img]
              y_batch += [mask]    

    
            x_batch = np.array(x_batch) / 255.
            y_batch = np.array(y_batch) / 255.

            yield x_batch, np.expand_dims(y_batch, -1)

def createAndTrainModel(trainImgPath, trainLabPath, testImgPath, testLabPath, valImgPath, valLabPath,
                        imageSize, learningRate, batchSize, epochs, modelName):

    model = models.unet_16_256([imageSize, imageSize, 3])
    model.summary()

    opt = tf.keras.optimizers.Adam(learning_rate=learningRate)

    model.compile(
      optimizer=opt,
      loss=models.soft_dice_loss,
      metrics=[models.iou_coef])

    if (not os.path.exists('./Models')):
        os.mkdir('./Models')

    checkpointer = ModelCheckpoint(os.path.join('./Models', modelName),
                             monitor="val_loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)

    earlystopper = EarlyStopping(monitor = 'val_loss', 
                          min_delta = 0, 
                          patience = 5,
                          verbose = 1,
                          restore_best_weights = True)

    lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=4,
                               verbose=1,
                               epsilon=1e-4)
                         
    history = model.fit_generator(train_generator(trainImgPath, trainLabPath, batchSize, [imageSize, imageSize, 3]),                              
                              steps_per_epoch=len(os.listdir(trainImgPath))/batchSize,
                              epochs=epochs,
                              verbose=1,
                              callbacks= [checkpointer, earlystopper, lr_reducer],
                              validation_data=train_generator(valImgPath, valLabPath, batchSize, [imageSize, imageSize, 3]),
                              validation_steps=1,
                              class_weight=None,
                              max_queue_size=10,
                              workers=1
                              )

    #model = load_model(os.path.join('./Models', modelName), custom_objects={'soft_dice_loss': models.soft_dice_loss, 'iou_coef': models.iou_coef})
    #model.evaluate_generator(train_generator(testImgPath, testLabPath, batchSize, [imageSize, imageSize, 3]), 
                            #steps=len(os.listdir(testImgPath))/batchSize
                            #)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--trainImages', type=str, default='./AugmentedDataset/train/images/', help="Images from train set.")
    parser.add_argument('--trainLabels', type=str, default='./AugmentedDataset/train/labels/', help="Labels from train set.")
    parser.add_argument('--valImages', type=str, default='./AugmentedDataset/val/images/', help="Images from validation set.")
    parser.add_argument('--valLabels', type=str, default='./AugmentedDataset/val/labels/', help="Labels from validation set.")
    parser.add_argument('--testImages', type=str, default='./AugmentedDataset/test/images/', help="Images from test set.")
    parser.add_argument('--testLabels', type=str, default='./AugmentedDataset/test/labels/', help="Labels from test set.")
    parser.add_argument('--imageSize', type=int, default=256, help="Image size.")
    parser.add_argument('--learningRate', type=float, default=0.0001, help="Learning rate.")
    parser.add_argument('--batchSize', type=int, default=64, help="Batch size.")
    parser.add_argument('--epochs', type=int, default=1, help="Number of epochs in training.")
    parser.add_argument('--modelName', type=str, default='model.h5', help="Name of the model.")

    args = parser.parse_args()

    createAndTrainModel(args.trainImages, args.trainLabels, args.testImages, args.testLabels, args.valImages, args.valLabels, 
                        args.imageSize, args.learningRate, args.batchSize, args.epochs, args.modelName)


if __name__ == "__main__":
    main()