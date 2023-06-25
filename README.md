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

## Inference 

For inference, use ```pred.py``` file to start testing.

The following command should be used:

```
python pred.py
```

Note: you should fix model_path for your model path and directory to your benchmark dataset.

## Pretrained weights

The weight for the PVTUNet will be in [Google Drive](https://drive.google.com/drive/folders/1vfzTBAsU28pNHuuYZ1L_zzok7sHi1Tn2?usp=sharing)

## Dataset

In our experiment, we use the dataset config from [PraNet](https://github.com/DengPingFan/PraNet), with training set from 50% of Kvasir-SEG and 50% of ClinicDB dataset. 

With our test dataset, we use the following:

In same distribution:

- Kvasir SEG

- ClinicDB 


Out of distribution:

- Etis dataset

- ColonDB

- CVC300

