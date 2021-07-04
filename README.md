# Roads-Segmentation
Roads Segmentation for Massachusetts Roads Dataset. Kaggle competition https://www.kaggle.com/balraj98/massachusetts-roads-dataset.
ML project for roads segmentation from aerial images, using Unet based models. Implementation in python using tensorflow and segmentation models library from https://github.com/qubvel/segmentation_models. This project was part of course Deep Learning, master studies on Faculty of Electronic Engineering, University of Niš.

Authors:
  Sreten Šikuljak
  Predrag Nikolić

Dataset is derived from kaggle competition. For purposes of this project all images and masks from train, val and test sets are combined into one dataset.
Dataset can be divided into train, test and val in any given proportions. 

Install dependencies:
```python
pip install -r req.txt
```

Result of project can be seen in *Data Visualization.ipynb*

There are three main scripts:
*deleteWhiteImages.py* - delete incomplete images that have complete masks, name of images are stored in imgList.txt
*config/deleteWhiteImages.yaml* 
```yaml
paths:
  dataset:
      images: path to dir with images
      labels: path to dir with masks
  delete_file: path to file with img names (imgList.txt)
```

*augmentDataset.py* - create dataset for traning model (create 256x256 tiles, augment dataset)
*config/augment.yaml* 
```yaml
paths:
  images: path to dir with images
  labels: path to dir with masks
  aug_path: path to dir with augmented dataset for training

split:
  train: 0-100 precentage of original dataset
  validation: 0-100 precentage of original dataset
  test: 0-100 precentage of original dataset

params:
  pixel_thd: 0-255 value for white pixel threshold
  tile_num: integer +=number of tiles derived from one image
  tile_size: integer tile size in pixels
```

*trainModel.py* - train model
*config/train.yaml* 
```yaml
paths:
  train:
      images: path to images train
      labels: path to masks train
  val:
      images: path to images val
      labels: path to masks val
  test:
      images: path to images test
      labels: path to masks test
  model: path to model in case of continuing training

model:
  backbone: pretrained model for encoder (models from segmentation lib:'resnet50', 'resnet34'...)
  encoder_weights: weights for pretrained model ('imagenet')
  encoder_freeze: freeze wights for pretrained model (True/False)
  decoder_filters: decoder number of filters in convolution ([256, 128, 64, 32, 16])

training:
  imageSize: input image size ([256, 256, 3])
  lr: 0.0001
  batchSize: 16
  epochs: 1
  load_model: load model for continue training (True/False)

outputs: (trainnig outputs)
  model: 'best_model.h5'
  log: 'log.csv'
  modelSummary: 'modelSummary.txt'
```
