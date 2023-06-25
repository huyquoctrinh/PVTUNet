# PVTUNet
This is the implementation for PVT UNet for segmentation

## Installation

Our implementation is on ``` Python 3.9 ``` , please make sure to config your environment compatible with the requirements.

To install all packages, use ``` requirements.txt ``` file to install. Install with ```pip ``` by the following command:

```
pip install -r requirements.txt
```

All packages will be automatically installed.


## Training

For training, use ``` train.py ``` file for start training.

The following command should be used:

```
python train.py
```

## Benchmark 

For benchmar, use ```test.py``` file for start testing.

The following command should be used:

```
python test.py
```

Note: you should fix model_path for your model path and directory to your benchmark dataset.

## Pretrained weights

The weight will be update later.

## Dataset

You can use Kvasir-SEG dataset for training, or CVC-clinic DB for training.

