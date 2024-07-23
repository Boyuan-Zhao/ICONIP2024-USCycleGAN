# Ultrasound enhance model

## Environment

Linux Nvidia-A800 with 1 GPU

## Installation

Use anaconda environment

```
conda create -n USenhance python=3.9
conda activate USenhance
pip install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Train

- Step1: Create dataset

  - split low quality, high quality images to A, B folder. (A: low quality, B: high quality)
  - train path/A/
  - train path/B/
  - val path/A/
  - val path/B/

  before training, you must modify `root_path` and `save_path` in `save_path.py` and run the following code to process data:
  ```
  python save_path.py
  ```

- Step2: Modify CycleGan.yaml options as follows

```
save_root: model save path
setname: train
dataroot: train path
val_dataroot: val path
```

- Step3: Run train

  please set the gpu-id when you are training
```
./train.sh 0
```

## validation
- Modify CycleGan.yaml options as follows

```
setname: val
model_root: your saved model path
```

- Run validation:

  please set the gpu-id when you are validation
```
./val.sh 0
```

## Use visdom：

```
python -m visdom.server -p 6022
```

If other port parameters are used, you need to modify the port in yaml.
