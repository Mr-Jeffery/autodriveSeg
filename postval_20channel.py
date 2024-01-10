import os
import torch
from PIL import Image
import numpy as np
# Define the root directory and the locations
root_dir = './output'
locations = ['berlin', 'bielefeld', 'bonn', 'leverkusen', 'mainz', 'munich']

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

assert len(ineval_src_labels) == len(ineval_tgt_labels), "src_labels and tgt_labels must have the same length"
post_train_trans = MapLabels(ineval_tgt_labels, ineval_src_labels)

# Iterate over each location
for location in locations:
    # Create a new directory for the location if it doesn't exist
    new_dir = os.path.join('./output2', location)
    os.makedirs(new_dir, exist_ok=True)
    # Iterate over each subdirectory in the root directory
    for subdir in os.listdir(root_dir):
        # Check if the subdirectory starts with the location name
        if subdir.startswith(location):
            # Define the path to the PNG file
            png_file = os.path.join(root_dir, subdir, subdir + '_seg.png')

            # Open the PNG file and convert it to a tensor
            img = Image.open(png_file)
            tensor = torch.from_numpy(np.array(img))

            # # Check the shape of the tensor
            # if tensor.shape != ( 2048, 1024):
            #     print(f"Skipping {png_file} due to unexpected shape {tensor.shape}")
            #     continue

            # # Permute the tensor
            # tensor = tensor.permute(1,0)
            
            tensor = post_train_trans(tensor)

            # Convert the tensor back to an image
            img = Image.fromarray(tensor.numpy().astype('uint8'))

            # Define the path to the new PNG file
            new_file = os.path.join(new_dir, subdir + '_seg.png')

            # Save the image
            img.save(new_file)
