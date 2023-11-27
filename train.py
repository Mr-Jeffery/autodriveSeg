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

import monai
from monai.data import ArrayDataset,  decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose,
    LoadImage,
    RandRotate90,
    RandSpatialCrop,
)
from monai.visualize import plot_2d_or_3d_image

parser = argparse.ArgumentParser()
# Add the --local_rank argument required by Accelerate
parser.add_argument("--max_epochs", default=10, type=int, help="max number of training epochs")# Jeffery modified
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--in_channels", default=3, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=34, type=int, help="number of output channels")
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
        ]
    )
    val_imtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), ])
    val_segtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), ])
    # # define array dataset, data loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create a training data loader
    train_ds = ArrayDataset(train_images, train_imtrans, train_segs, train_segtrans)# Jeffery modified
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=torch.cuda.is_available())

    # create a validation data loader
    val_ds = ArrayDataset(val_images, val_imtrans, val_segs, val_segtrans)# Jeffery modified
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, pin_memory=torch.cuda.is_available())
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    
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
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        dropout=1e-4
    ).to(device)

    if args.load_checkpoint is not None: # load existing check point
        model.load_state_dict(torch.load(args.load_checkpoint))

    loss_function = monai.losses.DiceLoss(sigmoid=True,to_onehot_y =True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)

    # start a typical PyTorch training
    val_interval = 2 # Validate every 2 training epoch
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter() # Add tensorboard writer
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
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

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
                    val_outputs = [tensor.argmax(axis=0,keepdim=True) for tensor in decollate_batch(val_outputs)]
                    dice_metric(y_pred=val_outputs, y=val_labels)
                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()
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
                plot_2d_or_3d_image(torch.permute(val_images[0],(0,2,1)), epoch + 1, writer, index=0, tag="image")
                plot_2d_or_3d_image(torch.permute(val_labels[0],(0,2,1)), epoch + 1, writer, index=0, tag="label")
                plot_2d_or_3d_image(torch.permute(val_outputs[0],(0,2,1)), epoch + 1, writer, index=0, tag="output")

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()
    exit()
    
if __name__ == "__main__":
    main()
