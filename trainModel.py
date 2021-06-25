from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.python.keras.models import load_model
import tensorflow as tf

import os

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

import utils 
import segmentation_models as sm
from segmentation_models import get_preprocessing

def createAndTrainModel(trainImgPath, trainLabPath, testImgPath, testLabPath, valImgPath, valLabPath, modelPath,
                        modelBackbone, encoderWeights, encoderFreeze, decoderFilters,
                        imageSize, learningRate, batchSize, epochs, loadTrainedModel, outputs):

    sm.set_framework('tf.keras')

    #model = utils.unet_16_256([imageSize, imageSize, 3])
    if loadTrainedModel:
      model = load_model(modelPath, custom_objects={'dice_loss': sm.losses.dice_loss, 'iou_score': sm.metrics.iou_score})
    else:
      model = sm.Unet(backbone_name=modelBackbone, 
                      input_shape=[None, None, 3], 
                      classes=1, 
                      activation='sigmoid',
                      encoder_weights=encoderWeights,
                      encoder_freeze=encoderFreeze,
                      decoder_filters=decoderFilters)

    preprocessInput = get_preprocessing(modelBackbone)

    with open(outputs.modelSummary, 'w') as f:
      model.summary(print_fn=lambda x: f.write(x + '\n'))

    opt = tf.keras.optimizers.Adam(learning_rate=learningRate)

    model.compile(
      optimizer=opt,
      loss=sm.losses.dice_loss,
      metrics=[sm.metrics.IOUScore(threshold=0.5)])

    logger = CSVLogger(outputs.log,
      separator=',',
      append=True
    )

    checkpointer = ModelCheckpoint(outputs.model,
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
                         
    history = model.fit(utils.generator(trainImgPath, trainLabPath, batchSize, preprocessInput),                              
                              steps_per_epoch=len(os.listdir(trainImgPath))/batchSize,
                              epochs=epochs,
                              verbose=1,
                              callbacks= [checkpointer, lr_reducer, logger],
                              validation_data=utils.generator(valImgPath, valLabPath, batchSize, preprocessInput),
                              validation_steps=len(os.listdir(valImgPath))/batchSize,
                              class_weight=None,
                              max_queue_size=10,
                              workers=1
                              )

    model = load_model(outputs.model, custom_objects={'dice_loss': sm.losses.dice_loss, 'iou_score': sm.metrics.iou_score})
    model.compile(
      optimizer=opt,
      loss=sm.losses.dice_loss,
      metrics=[sm.metrics.IOUScore(threshold=0.5)])
    
    test_gen = utils.generator(testImgPath, testLabPath, batchSize, preprocessInput)
    model.evaluate(
      test_gen,
      steps =len(os.listdir(testImgPath))/batchSize,
    )


@hydra.main(config_path='config', config_name='train.yaml')
def main(cfg: DictConfig):

    createAndTrainModel(
      to_absolute_path(cfg.paths.train.images),
      to_absolute_path(cfg.paths.train.labels),
      to_absolute_path(cfg.paths.test.images),
      to_absolute_path(cfg.paths.test.labels),
      to_absolute_path(cfg.paths.val.images),
      to_absolute_path(cfg.paths.val.labels), 
      to_absolute_path(cfg.paths.model), 
      cfg.model.backbone,
      cfg.model.encoder_weights,
      cfg.model.encoder_freeze,
      cfg.model.decoder_filters,
      cfg.training.imageSize[0],
      cfg.training.lr,
      cfg.training.batchSize,
      cfg.training.epochs,
      cfg.training.load_model,
      cfg.outputs
      )


if __name__ == "__main__":
    main()