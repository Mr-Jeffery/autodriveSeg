# autodriveSeg
Semantic segmentation for auto-driving using based on project MONAI.

## Environment

```
conda create -n seg
conda activate seg
conda install pip
conda install python=3.10
conda install numpy=1.26.0
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install monai==1.30
pip install tenosrboard
pip install tensorboardX==2.1
```
- Version Info
```
MONAI version: 1.3.0
Numpy version: 1.26.0
Pytorch version: 2.1.1
MONAI flags: HAS_EXT = False, USE_COMPILED = False, USE_META_DICT = False
MONAI rev id: 865972f7a791bf7b42efbcd87c8402bd865b329e
MONAI __file__: /work/<username>/miniconda3/envs/seg/lib/python3.10/site-packages/monai/__init__.py

Optional dependencies:
Pytorch Ignite version: NOT INSTALLED or UNKNOWN VERSION.
ITK version: NOT INSTALLED or UNKNOWN VERSION.
Nibabel version: NOT INSTALLED or UNKNOWN VERSION.
scikit-image version: NOT INSTALLED or UNKNOWN VERSION.
scipy version: 1.11.3
Pillow version: 10.0.1
Tensorboard version: 2.12.1
gdown version: NOT INSTALLED or UNKNOWN VERSION.
TorchVision version: 0.16.1
tqdm version: 4.59.0
lmdb version: NOT INSTALLED or UNKNOWN VERSION.
psutil version: 5.9.6
pandas version: NOT INSTALLED or UNKNOWN VERSION.
einops version: NOT INSTALLED or UNKNOWN VERSION.
transformers version: NOT INSTALLED or UNKNOWN VERSION.
mlflow version: NOT INSTALLED or UNKNOWN VERSION.
pynrrd version: NOT INSTALLED or UNKNOWN VERSION.
clearml version: NOT INSTALLED or UNKNOWN VERSION.
```

## Training
```
module load cuda/11.8
conda activate seg
python train.py \
    --max_epochs=500 \
    --batch_size=20 \
    --roi_x=512 \
    --roi_y=512 \
    --save_checkpoint="best_metric_model_segmentation2d_array_512by512_attention.pth" 2>&1 | tee tempoutput512_attention.txt
```
## Viewing Log
```
conda activate seg
tail -f tempoutput512_attention.txt
tensorboard --logdir=runs
```

## Evaluating
```
# Undone yet
```
