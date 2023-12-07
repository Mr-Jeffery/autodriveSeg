# autodriveSeg
Semantic segmentation for auto-driving using based on project MONAI.



## Labels
from [File Link]([/guides/content/editing-an-existing-page](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py))

| name                 	| id 	| trainId 	| category     	| categoryId 	| hasInstances 	| ignoreInEval 	| color           	|
|----------------------	|----	|---------	|--------------	|------------	|--------------	|--------------	|-----------------	|
| unlabeled            	| 0  	| 255     	| void         	| 0          	| False        	| True         	| (0, 0, 0)       	|
| ego vehicle          	| 1  	| 255     	| void         	| 0          	| False        	| True         	| (0, 0, 0)       	|
| rectification border 	| 2  	| 255     	| void         	| 0          	| False        	| True         	| (0, 0, 0)       	|
| out of roi           	| 3  	| 255     	| void         	| 0          	| False        	| True         	| (0, 0, 0)       	|
| static               	| 4  	| 255     	| void         	| 0          	| False        	| True         	| (0, 0, 0)       	|
| dynamic              	| 5  	| 255     	| void         	| 0          	| False        	| True         	| (111, 74, 0)    	|
| ground               	| 6  	| 255     	| void         	| 0          	| False        	| True         	| (81, 0, 81)     	|
| road                 	| 7  	| 0       	| flat         	| 1          	| False        	| False        	| (128, 64, 128)  	|
| sidewalk             	| 8  	| 1       	| flat         	| 1          	| False        	| False        	| (244, 35, 232)  	|
| parking              	| 9  	| 255     	| flat         	| 1          	| False        	| True         	| (250, 170, 160) 	|
| rail track           	| 10 	| 255     	| flat         	| 1          	| False        	| True         	| (230, 150, 140) 	|
| building             	| 11 	| 2       	| construction 	| 2          	| False        	| False        	| (70, 70, 70)    	|
| wall                 	| 12 	| 3       	| construction 	| 2          	| False        	| False        	| (102, 102, 156) 	|
| fence                	| 13 	| 4       	| construction 	| 2          	| False        	| False        	| (190, 153, 153) 	|
| guard rail           	| 14 	| 255     	| construction 	| 2          	| False        	| True         	| (180, 165, 180) 	|
| bridge               	| 15 	| 255     	| construction 	| 2          	| False        	| True         	| (150, 100, 100) 	|
| tunnel               	| 16 	| 255     	| construction 	| 2          	| False        	| True         	| (150, 120, 90)  	|
| pole                 	| 17 	| 5       	| object       	| 3          	| False        	| False        	| (153, 153, 153) 	|
| polegroup            	| 18 	| 255     	| object       	| 3          	| False        	| True         	| (153, 153, 153) 	|
| traffic light        	| 19 	| 6       	| object       	| 3          	| False        	| False        	| (250, 170, 30)  	|
| traffic sign         	| 20 	| 7       	| object       	| 3          	| False        	| False        	| (220, 220, 0)   	|
| vegetation           	| 21 	| 8       	| nature       	| 4          	| False        	| False        	| (107, 142, 35)  	|
| terrain              	| 22 	| 9       	| nature       	| 4          	| False        	| False        	| (152, 251, 152) 	|
| sky                  	| 23 	| 10      	| sky          	| 5          	| False        	| False        	| (70, 130, 180)  	|
| person               	| 24 	| 11      	| human        	| 6          	| True         	| False        	| (220, 20, 60)   	|
| rider                	| 25 	| 12      	| human        	| 6          	| True         	| False        	| (255, 0, 0)     	|
| car                  	| 26 	| 13      	| vehicle      	| 7          	| True         	| False        	| (0, 0, 142)     	|
| truck                	| 27 	| 14      	| vehicle      	| 7          	| True         	| False        	| (0, 0, 70)      	|
| bus                  	| 28 	| 15      	| vehicle      	| 7          	| True         	| False        	| (0, 60, 100)    	|
| caravan              	| 29 	| 255     	| vehicle      	| 7          	| True         	| True         	| (0, 0, 90)      	|
| trailer              	| 30 	| 255     	| vehicle      	| 7          	| True         	| True         	| (0, 0, 110)     	|
| train                	| 31 	| 16      	| vehicle      	| 7          	| True         	| False        	| (0, 80, 100)    	|
| motorcycle           	| 32 	| 17      	| vehicle      	| 7          	| True         	| False        	| (0, 0, 230)     	|
| bicycle              	| 33 	| 18      	| vehicle      	| 7          	| True         	| False        	| (119, 11, 32)   	|
| license plate        	| -1 	| -1      	| vehicle      	| 7          	| False        	| True         	| (0, 0, 142)     	|

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
model converges in about 100 epoches when the batch size is set to 1 and roi is set to 2048*1024.
```
module load cuda/11.8
conda activate seg
python train.py \
    --max_epochs=100 \
    --batch_size=1 \
    --roi_x=2048 \
    --roi_y=1024 \
    --save_checkpoint="best_metric_model_segmentation2d_array.pth" 2>&1 | tee tempoutput.txt
```
## Viewing Log
```
conda activate seg
tail -f tempoutput_attention.txt
tensorboard --logdir=runs
```

## Evaluating
```
module load cuda/11.8
conda activate seg
python eval.py \
    --batch_size=1 \
    --roi_x=2048 \
    --roi_y=1024 \
    --load_checkpoint="best_metric_model_segmentation2d_array.pth" 2>&1 | tee tempoutput.txt
```
## Research Design and Methods

The main code is based on [unet_training from project MONAI](https://github.com/Project-MONAI/tutorials/blob/main/2d_segmentation/torch/unet_training_array.py). MONAI is a PyTorch-based, open-source framework for deep learning in healthcare imaging, part of the PyTorch Ecosystem. This package has implemented many different neural network for semenstic segmentation, making it conenient to deploy and modify.

For model, Feiyang suggests the usage of [AttentionUnet](https://docs.monai.io/en/stable/_modules/monai/networks/nets/attentionunet.html), this model is a modified version of Unet with **Attention Gate** incorporated  inside. Gradients originating from background regions are down weighted
during the backward pass. This allows model parameters in shallower layers to be updated mostly based on spatial regions that are relevant to a given task. Additive attention is formulated as follows:
$$
q^l_{att} =  \psi^T (\sigma_1 (W^T_x x^l_i + W^T_g g^l_i + b_g)) +b_{\psi}
\alpha^l_i  =  \sigma_2 (q^l_{att} (x^l_i, g_i ;\Theta_{att}))
$$
where $\sigma_2 = \frac{1}{\exp (- x_{i, c})}$

 AG is characterised by a set of parameters $\Theta_{att}$ containing: linear transformations $W_x \in
\mathbb{R}^{F_l \times F_{int}}, W_g \in \mathbb{R}^{F_g \times
F_{int}}, \psi \in \mathbb{R}^{F_{int} \times 1}$

The original paper proposed a grid-attention technique. In this case, gating signal is not a global single vector for all image pixels but a grid signal conditioned to image spatial information.

We have used the Cityscapes dataset to train the model which contained RGB images along with their corresponding finely annotated images for semantic segmentation. There were a total of 5000 images which were further divided into training, validation and test sets. These input images and segmentation masks were resized from the original resolution of 1024 $\times$ 2048 to 512 $\times$ 512 in order to decrease the training time while causing negligible loss of information. To keep the model unbiased to the training images, we have shuffled the training images before feeding them for training. 

##  Initial Results
Our First submission uses Attention Unet with 512 $\times$ 512 sliding window inference. The submission returns the following result:
| Metric          	| Value   	|
|-----------------	|---------	|
| IoU Classes     	| 45.2974 	|
| iIoU Classes    	| 29.3292 	|
| IoU Categories  	| 79.5781 	|
| iIoU Categories 	| 70.4729 	|

| Class         	| IoU       	| iIoU      	|
|---------------	|-----------	|-----------	|
| road          	| 94.1033   	| -         	|
| sidewalk      	| 31.5197   	| -         	|
| building      	| 81.4073   	| -         	|
| wall          	| 17.5878   	| -         	|
| fence         	| 24.1842   	| -         	|
| pole          	| 49.2604   	| -         	|
| traffic light 	| 13.1106   	| -         	|
| traffic sign  	| 62.9409   	| -         	|
| vegetation    	| 89.3674   	| -         	|
| terrain       	| 60.3792   	| -         	|
| sky           	| 89.25     	| -         	|
| person        	| 65.4055   	| 56.0844   	|
| rider         	| 18.0407   	| 21.1663   	|
| car           	| 85.2847   	| 79.8377   	|
| truck         	| 0.0564402 	| 0.0893015 	|
| bus           	| 3.22736   	| 6.22195   	|
| train         	| 6.24262   	| 7.22441   	|
| motorcycle    	| 15.88     	| 13.3991   	|
| bicycle       	| 53.4023   	| 50.6106   	|

The accuracy is quite low compare to the existing benchmark:
| Metric          	| Value   	|
|-----------------	|---------	|
| IoU Classes     	| 85.8667 	|
| iIoU Classes    	| 69.0301 	|
| IoU Categories  	| 93.1585 	|
| iIoU Categories 	| 84.7639 	|

| Class         	| IoU     	| iIoU    	|
|---------------	|---------	|---------	|
| road          	| 98.9975 	| -       	|
| sidewalk      	| 89.4421 	| -       	|
| building      	| 94.881  	| -       	|
| wall          	| 73.126  	| -       	|
| fence         	| 69.1677 	| -       	|
| pole          	| 75.6851 	| -       	|
| traffic light 	| 82.1686 	| -       	|
| traffic sign  	| 85.0499 	| -       	|
| vegetation    	| 94.4899 	| -       	|
| terrain       	| 75.8745 	| -       	|
| sky           	| 96.3034 	| -       	|
| person        	| 90.0583 	| 77.4795 	|
| rider         	| 79.3188 	| 63.2564 	|
| car           	| 97.0099 	| 92.3638 	|
| truck         	| 83.7241 	| 48.1885 	|
| bus           	| 94.9399 	| 73.9745 	|
| train         	| 92.3479 	| 63.456  	|
| motorcycle    	| 77.3525 	| 61.618  	|
| bicycle       	| 81.5298 	| 71.904  	|
