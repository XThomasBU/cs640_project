import os
import shutil

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import glob
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


train_df = pd.read_csv("/projectnb/cs640grp/materials/UBC-OCEAN_CS640/train.csv")
test_df = pd.read_csv("/projectnb/cs640grp/materials/UBC-OCEAN_CS640/test.csv")


BASE_DIR = ["/projectnb/cs640grp/materials/UBC-OCEAN_CS640/", "/projectnb/cs640grp/materials/UBC-OCEAN_CS640/"]
TRAIN_DIR= "/projectnb/cs640grp/materials/UBC-OCEAN_CS640/train_images_compressed_80"


def organize_images_by_label(df: pd.DataFrame, source_dir: str) -> None:
    image_paths=[]
    for _, row in df.iterrows():
        image_id = row["image_id"]
        label = row["label"]
        source_path = os.path.join(source_dir, f"{image_id}.jpg") 
        try:
            image_paths.append(os.path.join(source_dir, f"{image_id}.jpg"))
        except FileNotFoundError:
            image_paths.append(1)
            continue
    return image_paths


image_paths = organize_images_by_label(train_df, BASE_DIR[0])
train_df['image_path'] = image_paths

# shuffle train_df
train_df = train_df.sample(frac=1).reset_index(drop=True)

print(train_df.head())

# Assuming df is your DataFrame
unique_labels = train_df['label'].unique()

# Create a list to store one sample from each unique label
samples_list = [train_df[train_df['label'] == label].iloc[0] for label in unique_labels]

# Create a new DataFrame from the list of samples
samples_df = pd.DataFrame(samples_list)

# Display the resulting DataFrame
print(samples_df)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

for (idx, label) in zip(samples_df['image_id'].values, samples_df['label'].values):
    pacthes_dir = f'/projectnb/cs640grp/students/xthomas/FINAL_PROJECT/patches/{idx}'
    pacthes = os.listdir(pacthes_dir)
    patches_paths = [os.path.join(pacthes_dir, patch) for patch in pacthes]

    # Create subplots for each patch in a row
    fig, axes = plt.subplots(1, len(patches_paths), figsize=(15, 5))

    for i, patch_path in enumerate(patches_paths):
        img = mpimg.imread(patch_path)
        axes[i].imshow(img)
        axes[i].axis('off')

    # Set the title for the row based on label
    plt.suptitle(f"Label: {label}")

    plt.savefig(f'patches_plots/patches_{label}.png')
    