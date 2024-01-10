import logging
import os
import sys
import argparse
from glob import glob

import torch

import monai
from monai import config
from monai.data import ArrayDataset, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import Compose, LoadImage, SaveImage

parser = argparse.ArgumentParser()
# Add the --local_rank argument required by Accelerate
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--in_channels", default=3, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=20, type=int, help="number of output channels")
parser.add_argument("--load_checkpoint", default="best_metric_model_segmentation2d_array.pth", type=str, help="model")
args = parser.parse_args()

def main():
    config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # print(f"generating synthetic data to {dir} (this may take a while)")

    imdir = '/work/APAC-TY/feiyeung/cs405/segment/leftImg8bit/test/*'
    images = sorted(glob(os.path.join(imdir, "*_leftImg8bit.png")))
    segdir = '/work/APAC-TY/feiyeung/cs405/segment/gtFine/test/*'
    segs = sorted(glob(os.path.join(segdir, "*_gtFine_labelIds.png")))

    # define transforms for image and segmentation
    imtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True)])
    segtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True) ])
    val_ds = ArrayDataset(images, imtrans, segs, segtrans)
    # sliding window inference for one image at every iteration
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    saver = SaveImage(output_dir="./output", output_ext=".png", output_postfix="seg")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.AttentionUnet(
        spatial_dims=2,
        in_channels=args.in_channels,# Jeffery modified
        out_channels=args.out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(4, 4, 4, 2),
        dropout=0
    ).to(device)

    model.load_state_dict(torch.load(args.load_checkpoint))
    model.eval()
    with torch.no_grad():
        for val_data in val_loader:
            val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
            # define sliding window size and batch size for windows inference
            roi_size = (args.roi_x, args.roi_y)
            sw_batch_size = args.sw_batch_size
            val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
            val_outputs = [tensor.argmax(axis=0,keepdim=True) for tensor in decollate_batch(val_outputs)]
            # val_labels = decollate_batch(val_labels)
            # compute metric for current iteration
            dice_metric(y_pred=val_outputs, y=val_labels)
            for val_output in val_outputs:
                saver(val_output)
        # aggregate the final mean dice result
        print("evaluation metric:", dice_metric.aggregate().item())
        # reset the status
        dice_metric.reset()


if __name__ == "__main__":
    main()