# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified from https://github.com/Project-MONAI/tutorials/blob/main/2d_segmentation/torch/unet_training_array.py

import logging
import argparse
import os
import sys
from glob import glob

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.cuda import memory_allocated

import monai
from monai.data import ArrayDataset,  decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, MeanIoU
from monai.transforms import (
    Compose,
    LoadImage,
    RandRotate90,
    RandSpatialCrop,
    ToTensor,
)
from monai.visualize import plot_2d_or_3d_image

parser = argparse.ArgumentParser()
# Add the --local_rank argument required by Accelerate
parser.add_argument("--max_epochs", default=10, type=int, help="max number of training epochs")# Jeffery modified
parser.add_argument("--batch_size", default=4, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--in_channels", default=3, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=20, type=int, help="number of output channels")
parser.add_argument("--save_checkpoint", default="best_metric_model_segmentation2d_array.pth", type=str, help="model")
parser.add_argument("--load_checkpoint", default=None, type=str, help="model")
args = parser.parse_args()

def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    train_imdir = '/work/APAC-TY/feiyeung/cs405/segment/leftImg8bit/train/*'# Training Images Directory
    train_images = sorted(glob(os.path.join(train_imdir, "*_leftImg8bit.png")))
    train_segdir = '/work/APAC-TY/feiyeung/cs405/segment/gtFine/train/*'# Training Labels/Segmented Images Directory
    train_segs = sorted(glob(os.path.join(train_segdir, "*_gtFine_labelIds.png")))# Suffix
    print('train examples num: '+str(len(train_images)))

    val_imdir = '/work/APAC-TY/feiyeung/cs405/segment/leftImg8bit/val/*'# Validation Images Directory
    val_images = sorted(glob(os.path.join(val_imdir, "*_leftImg8bit.png")))
    val_segdir = '/work/APAC-TY/feiyeung/cs405/segment/gtFine/val/*'# Validation Labels/Segmented Images Directory
    val_segs = sorted(glob(os.path.join(val_segdir, "*_gtFine_labelIds.png")))# Suffix

    print('validate examples num: '+str(len(val_images)))
    # define transforms for image and segmentation

    class MapLabels:
        def __init__(self, src_labels, tgt_labels):
            assert isinstance(src_labels, (list, tuple)), "src_labels must be a list or tuple"
            assert isinstance(tgt_labels, (list, tuple)), "tgt_labels must be a list or tuple"
            self.src_labels = src_labels
            self.tgt_labels = tgt_labels

        def __call__(self, label):
            label_copy = label.clone().float()  # Create a copy and convert to FloatTensor
            for src_label, tgt_label in zip(self.src_labels, self.tgt_labels):
                label_copy[label == src_label] = tgt_label
            return label_copy.long()  # Convert back to LongTensor


    ineval_src_labels = [7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33]
    ineval_tgt_labels = list(range(1, 20))
    uneval_src_labels = [0,1,2,3,4,5,6,9,10,14,15,16,18,29,30]
    uneval_tgt_labels = [0] * len(uneval_src_labels)

    assert len(ineval_src_labels) == len(ineval_tgt_labels), "src_labels and tgt_labels must have the same length"
        
    train_imtrans = Compose(# The values in the set would be range(256)
        [
            LoadImage(image_only=True, ensure_channel_first=True),
            RandSpatialCrop((args.roi_x, args.roi_y), random_size=False),# Jeffery modified
            RandRotate90(prob=0, spatial_axes=(0, 1)),
        ]
    )
    train_segtrans = Compose(# The values in the set would be range(34)
        [
            LoadImage(image_only=True, ensure_channel_first=True),
            RandSpatialCrop((args.roi_x, args.roi_y), random_size=False),# Jeffery modified
            RandRotate90(prob=0, spatial_axes=(0, 1)),
            ToTensor(dtype=torch.int),
            MapLabels(uneval_src_labels, uneval_tgt_labels),
            MapLabels(ineval_src_labels, ineval_tgt_labels), 
        ]
    )
    val_imtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), ])
    val_segtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True),
                            ToTensor(dtype=torch.int),
                            MapLabels(uneval_src_labels, uneval_tgt_labels),
                            MapLabels(ineval_src_labels, ineval_tgt_labels),
                            ])
    
    # # define array dataset, data loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create a training data loader
    train_ds = ArrayDataset(train_images, train_imtrans, train_segs, train_segtrans)# Jeffery modified
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=torch.cuda.is_available())

    post_train_trans = MapLabels(ineval_tgt_labels, ineval_src_labels)

    # create a validation data loader
    val_ds = ArrayDataset(val_images, val_imtrans, val_segs, val_segtrans)# Jeffery modified
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())

    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    iou_metric = MeanIoU(include_background=False, reduction="mean", get_not_nans=False, ignore_empty=True)
    
    # create UNet, DiceLoss and Adam optimizer
 
    # model = monai.networks.nets.UNet(# Alternatively you could use Unet, but the outcome is not satisfying
    #     spatial_dims=2,
    #     in_channels=args.in_channels,# Jeffery modified
    #     out_channels=args.out_channels,
    #     channels=(16, 32, 64, 128, 256),
    #     strides=(2, 2, 2, 2),
    #     num_res_units=4,
    # ).to(device)

    model = monai.networks.nets.AttentionUnet(
        spatial_dims=2,
        in_channels=args.in_channels,# Jeffery modified
        out_channels=args.out_channels,
        channels=(16, 32, 64, 128, 256),#(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        dropout=1e-4
    ).to(device)

    if args.load_checkpoint is not None: # load existing check point
        model.load_state_dict(torch.load(args.load_checkpoint))

    # loss_function = monai.losses.DiceLoss(sigmoid=True,to_onehot_y =True)
    loss_weight = [0.2,0.8,0.6,1,1,0.9,1,0.9,0.2,0.8,0.2,0.7,1,0.5,1,1,1,1,0.8]
    loss_function = monai.losses.DiceFocalLoss(sigmoid=True,to_onehot_y =True,weight=loss_weight,include_background=False)

    optimizer = torch.optim.Adam(model.parameters(), 1e-3)

    # start a typical PyTorch training
    val_interval = 1 # Validate every ? training epoch
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter() # Add tensorboard writer

    # prof = torch.profiler.profile(
    #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./runs'),
    #     record_shapes=True,
    #     with_stack=True)

    # prof.start()

    # Define the dictionary of id to color
    def id2color(tensor):
        id_to_color = {
        0: (0, 0, 0),
        1: (0, 0, 0),
        2: (0, 0, 0),
        3: (0, 0, 0),
        4: (0, 0, 0),
        5: (111, 74, 0),
        6: (81, 0, 81),
        7: (128, 64, 128),
        8: (244, 35, 232),
        9: (250, 170, 160),
        10: (230, 150, 140),
        11: (70, 70, 70),
        12: (102, 102, 156),
        13: (190, 153, 153),
        14: (180, 165, 180),
        15: (150, 100, 100),
        16: (150, 120, 90),
        17: (153, 153, 153),
        18: (153, 153, 153),
        19: (250, 170, 30),
        20: (220, 220, 0),
        21: (107, 142, 35),
        22: (152, 251, 152),
        23: (70, 130, 180),
        24: (220, 20, 60),
        25: (255, 0, 0),
        26: (0, 0, 142),
        27: (0, 0, 70),
        28: (0, 60, 100),
        29: (0, 0, 90),
        30: (0, 0, 110),
        31: (0, 80, 100),
        32: (0, 0, 230),
        33: (119, 11, 32),
        }
        rgb_tensor = torch.zeros(tensor.shape[0], 3, tensor.shape[2], tensor.shape[3])
        for key, value in id_to_color.items():
                    mask = tensor[0,0,:,:] == key
                    rgb_tensor[0, :, mask] = torch.tensor(value, dtype=torch.float32).view(3, 1)
        return rgb_tensor

    for epoch in range(args.max_epochs):
        print("-" * 10)
        print(f"epoch {epoch+1}/{args.max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:# Update Loss
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

            # New code to record CUDA memory usage
            cuda_memory = memory_allocated(device)
            writer.add_scalar("cuda_memory", cuda_memory, epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        print(torch.cuda.get_device_name(0))
        print(torch.cuda.memory_allocated(0))

        if (epoch + 1) % val_interval == 0:# validation
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                for val_data in val_loader:
                    val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                    roi_size = (args.roi_x, args.roi_y)
                    sw_batch_size = args.sw_batch_size
                    val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                    val_outputs = torch.stack([tensor.argmax(axis=0,keepdim=True) for tensor in decollate_batch(val_outputs)])
                    # val_outputs = val_outputs.argmax(axis=0,keepdim=True)
                    dice_metric(y_pred=val_outputs, y=val_labels)
                    # iou_metric(y_pred=val_outputs, y=val_labels)
                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # metric = iou_metric.aggregate().item()

                # reset the status for next validation round
                dice_metric.reset()
                # iou_metric.reset()
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), args.save_checkpoint)
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_mean_dice", metric, epoch + 1)
                # plot the first model output as GIF image in TensorBoard with the corresponding image and label
                # writer.add_images('image', val_images.permute(0,1,3,2)/255, 0)
                # writer.add_images('label', id2color(val_images).permute(0,1,3,2)/255, 0)
                # writer.add_images('output', id2color(val_outputs).permute(0,1,3,2)/255, 0)
                print(torch.unique(post_train_trans(val_labels).view(-1)))
                print(torch.unique(post_train_trans(val_outputs).view(-1)))
                plot_2d_or_3d_image(val_images.permute(0,1,3,2)/255, 1, writer, index=0, tag="data", max_channels=3)
                plot_2d_or_3d_image(id2color(tensor=post_train_trans(val_labels)).permute(0,1,3,2)/255, epoch + 1, writer, index=0, tag="label", max_channels=3)
                print(val_outputs)
                plot_2d_or_3d_image(id2color(tensor=post_train_trans(val_outputs)).permute(0,1,3,2)/255, epoch + 1, writer, index=0, tag="output", max_channels=3)
                # def confusionMat(y_pred, y):
                #     import io
                #     import PIL
                #     import matplotlib.pyplot as plt
                #     from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
                #     cm = confusion_matrix(y_pred, y)
                #     disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                #     disp.plot()
                #     plt.title("Confusion Matrix")
                #     buf = io.BytesIO()
                #     plt.savefig(buf, format='jpeg')
                #     buf.seek(0)
                #     image = PIL.Image.open(buf)
                #     image = ToTensor()(image).unsqueeze(0)
                #     return image
                # writer.add_images('image', confusionMat(y_pred, y), 0)                
                


    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    # prof.stop()
    writer.close()
    exit()
    
if __name__ == "__main__":
    main()
