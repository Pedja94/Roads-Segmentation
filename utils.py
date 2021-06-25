from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers.core import Dropout, Lambda
from tensorflow.python.keras.layers.convolutional import Conv2D, Conv2DTranspose
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.python.keras.layers.merge import concatenate
from tensorflow.python.keras.layers import BatchNormalization
import tensorflow.keras.applications as tf_app 
from keras_applications import imagenet_utils
from keras import backend as K
import numpy as np
import cv2
import os

def get_preprocess_fn(backboneName):

  fn_dict = {
    'resnet18': tf_app.imagenet_utils.preprocess_input,
    'resnet34': tf_app.imagenet_utils.preprocess_input,
    'resnet50': tf_app.resnet50.preprocess_input,
    'vgg16': tf_app.vgg16.preprocess_input,
    'inceptionv3': tf_app.inception_v3.preprocess_input,
    'inceptionresnetv2': tf_app.inception_resnet_v2.preprocess_input
  }

  return fn_dict[backboneName]


def load_data(imgPath, labPath):

  images = []
  masks = []

  for imgName in os.listdir(imgPath):
    images.append(cv2.imread(os.path.join(imgPath, imgName)))
    masks.append(cv2.imread(os.path.join(labPath, imgName)))

  return np.array(images), np.array(masks)


def generator(img_dir, label_dir, batch_size, preproces_fn = None):
    list_images = os.listdir(img_dir)
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
          cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
          mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)      

          x_batch += [img]
          y_batch += [mask]    

        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch) / 255.

        if (preproces_fn is not None):
          x_batch = preproces_fn(x_batch)
        else:
          x_batch = x_batch / 255.

        yield x_batch, np.expand_dims(y_batch, -1)


def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  
  return iou


def dice_coef(y_true, y_pred, smooth = 1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def soft_dice_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def unet_16_256(imgSize):
    inputs = Input(imgSize)

    conv1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (inputs)
    conv1 = BatchNormalization() (conv1)
    conv1 = Dropout(0.1) (conv1)
    conv1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (conv1)
    conv1 = BatchNormalization() (conv1)
    pooling1 = MaxPooling2D((2, 2)) (conv1)

    conv2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (pooling1)
    conv2 = BatchNormalization() (conv2)
    conv2 = Dropout(0.1) (conv2)
    conv2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (conv2)
    conv2 = BatchNormalization() (conv2)
    pooling2 = MaxPooling2D((2, 2)) (conv2)

    conv3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (pooling2)
    conv3 = BatchNormalization() (conv3)
    conv3 = Dropout(0.2) (conv3)
    conv3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (conv3)
    conv3 = BatchNormalization() (conv3)
    pooling3 = MaxPooling2D((2, 2)) (conv3)

    conv4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (pooling3)
    conv4 = BatchNormalization() (conv4)
    conv4 = Dropout(0.2) (conv4)
    conv4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (conv4)
    conv4 = BatchNormalization() (conv4)
    pooling4 = MaxPooling2D(pool_size=(2, 2)) (conv4)

    conv5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (pooling4)
    conv5 = BatchNormalization() (conv5)
    conv5 = Dropout(0.3) (conv5)
    conv5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (conv5)
    conv5 = BatchNormalization() (conv5)


    upsample6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (conv5)
    upsample6 = concatenate([upsample6, conv4])
    conv6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (upsample6)
    conv6 = BatchNormalization() (conv6)
    conv6 = Dropout(0.2) (conv6)
    conv6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (conv6)
    conv6 = BatchNormalization() (conv6)

    upsample7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (conv6)
    upsample7 = concatenate([upsample7, conv3])
    conv7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (upsample7)
    conv7 = BatchNormalization() (conv7)
    conv7 = Dropout(0.2) (conv7)
    conv7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (conv7)
    conv7 = BatchNormalization() (conv7)

    upsample8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (conv7)
    upsample8 = concatenate([upsample8, conv2])
    conv8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (upsample8)
    conv8 = BatchNormalization() (conv8)
    conv8 = Dropout(0.1) (conv8)
    conv8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (conv8)
    conv8 = BatchNormalization() (conv8)

    upsample9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (conv8)
    upsample9 = concatenate([upsample9, conv1], axis=3)
    conv9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (upsample9)
    conv9 = BatchNormalization() (conv9)
    conv9 = Dropout(0.1) (conv9)
    conv9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (conv9)
    conv9 = BatchNormalization() (conv9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (conv9)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model


def unet_64_512(imgSize):
    inputs = Input(imgSize)

    conv1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (inputs)
    conv1 = BatchNormalization() (conv1)
    conv1 = Dropout(0.1) (conv1)
    conv1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (conv1)
    conv1 = BatchNormalization() (conv1)
    pooling1 = MaxPooling2D((2, 2)) (conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (pooling1)
    conv2 = BatchNormalization() (conv2)
    conv2 = Dropout(0.1) (conv2)
    conv2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (conv2)
    conv2 = BatchNormalization() (conv2)
    pooling2 = MaxPooling2D((2, 2)) (conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (pooling2)
    conv3 = BatchNormalization() (conv3)
    conv3 = Dropout(0.2) (conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (conv3)
    conv3 = BatchNormalization() (conv3)
    pooling3 = MaxPooling2D((2, 2)) (conv3)

    conv4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (pooling3)
    conv4 = BatchNormalization() (conv4)
    conv4 = Dropout(0.2) (conv4)
    conv4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (conv4)
    conv4 = BatchNormalization() (conv4)


    upsample5 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (conv4)
    upsample5 = concatenate([upsample5, conv3])
    conv5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (upsample5)
    conv5 = BatchNormalization() (conv5)
    conv5 = Dropout(0.2) (conv5)
    conv5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (conv5)
    conv5 = BatchNormalization() (conv5)

    upsample6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (conv5)
    upsample6 = concatenate([upsample6, conv2])
    conv6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (upsample6)
    conv6 = BatchNormalization() (conv6)
    conv6 = Dropout(0.2) (conv6)
    conv6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (conv6)
    conv6 = BatchNormalization() (conv6)

    upsample7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (conv6)
    upsample7 = concatenate([upsample7, conv1])
    conv7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (upsample7)
    conv7 = BatchNormalization() (conv7)
    conv7 = Dropout(0.1) (conv7)
    conv7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (conv7)
    conv7 = BatchNormalization() (conv7)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (conv7)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model