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


import cv2
import numpy as np

def is_informative(patch, black_threshold=0.05, white_threshold=0.05, std_threshold=25, unique_pixels_threshold=30):
    """
    Check if a patch is informative based on black and white percentages, standard deviation, and unique pixels.

    Parameters:
    - patch (numpy array): The image patch.
    - black_threshold (float): The maximum black percentage for a patch to be considered informative.
    - white_threshold (float): The maximum white percentage for a patch to be considered informative.
    - std_threshold (float): The minimum standard deviation for a patch to be considered informative.
    - unique_pixels_threshold (int): The minimum number of unique pixel values for a patch to be considered informative.

    Returns:
    - bool: True if the patch is informative, False otherwise.
    """
    # Convert the patch to grayscale for simplicity
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    # Calculate the percentage of black pixels
    black_percentage = np.sum(gray == 0) / gray.size

    # Calculate the percentage of white pixels
    white_percentage = np.sum(gray == 255) / gray.size


    black_and_white_percentage = np.sum(gray == 0) / gray.size + np.sum(gray == 255) / gray.size

    if black_and_white_percentage >= 0.05:
        return False

    # Check if the black and white percentages are below the thresholds
    if black_percentage >= black_threshold or white_percentage >= white_threshold:
        return False

    # Check if the standard deviation and unique pixels conditions are met
    return np.std(gray) > std_threshold and len(np.unique(np.int16(gray))) > unique_pixels_threshold


def extract_informative_patches(image_path, patch_size=(256, 256), num_patches=10):
    """
    Extract informative patches from an image.
    
    Parameters:
    - image_path (str): Path to the input image.
    - patch_size (tuple): Size of the patches to extract. Default is (256, 256).
    - num_patches (int): Number of random patches to extract. Default is 10.

    Returns:
    - list: List of extracted informative patches.
    """

    # Load the image
    image = np.array(Image.open(image_path))

    # Check if image loaded correctly
    if image is None:
        raise ValueError("Could not load the image from the provided path.")

    # Image dimensions
    img_height, img_width = image.shape[:2]

    # Ensure the image is large enough for the patch size
    if img_width < patch_size[0] or img_height < patch_size[1]:
        raise ValueError("Image dimensions are smaller than the patch size.")

    patches = []
    attempts = 0
    max_attempts = num_patches * 100  # Max number of tries to extract informative patches

    while len(patches) < num_patches:
        # Randomly select top-left corner of the patch
        x = np.random.randint(0, img_width - patch_size[0])
        y = np.random.randint(0, img_height - patch_size[1])

        # Extract patch
        patch = image[y:y+patch_size[1], x:x+patch_size[0]]
        
        # Check if the patch is informative
        if is_informative(patch):
            patches.append(patch)

        attempts += 1
    return patches

def zoom_and_crop(image, zoom_factor=1):
    # Zoom (resize) the image
    h, w = image.shape[:2]
    enlarged = cv2.resize(image, (w*zoom_factor, h*zoom_factor), interpolation=cv2.INTER_LINEAR)

    # Crop the center
    center_x, center_y = enlarged.shape[1] // 2, enlarged.shape[0] // 2
    left_x, right_x = center_x - w//2, center_x + w//2
    top_y, bottom_y = center_y - h//2, center_y + h//2
    
    cropped = enlarged[top_y:bottom_y, left_x:right_x]
    return cropped

def plot_patches(patches):
    fig, axes = plt.subplots(1, len(patches), figsize=(20, 5))
    for ax, patch in zip(axes, patches):
        ax.imshow(patch)
        ax.axis('off')
    plt.savefig("patches.png")
    
def random_centered_crop(image_path, patch_size=512, num_patches=10):
    """
    Generate random patches centered around the image center.

    Args:
    - image_path (str): Input image path.
    - patch_size (int): Size of the side for square cropping. Default is 256.
    - num_patches (int): Number of random patches to generate. Default is 10.
    
    Returns:
    - list: List of randomly cropped patches.
    """
    image = np.array(Image.open(image_path))
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2
    patches = []

    for _ in range(num_patches):
        # Generate random offsets
        offset_x = np.random.randint(-patch_size//2, patch_size//2)
        offset_y = np.random.randint(-patch_size//2, patch_size//2)
        startx = center_x + offset_x - patch_size // 2
        starty = center_y + offset_y - patch_size // 2
        endx = startx + patch_size
        endy = starty + patch_size

        # Handle cases where the crop goes out of image boundaries
        startx = max(0, startx)
        starty = max(0, starty)
        endx = min(width, endx)
        endy = min(height, endy)

        patch = image[starty:endy, startx:endx]
        patch = cv2.resize(patch, (256, 256), interpolation=cv2.INTER_LINEAR)
        patches.append(patch)

    return patches


def resize(image):
    resized_img = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
    return resized_img

def save_patches(patches, image_id):
    source_dir="/projectnb/cs640grp/students/xthomas/FINAL_PROJECT/patches/"+str(image_id)
    if not os.path.isdir(source_dir):
        os.makedirs(source_dir)
    for idx, patch in enumerate(patches):
        filename = os.path.join(source_dir, f"{image_id}_patch_{idx}.jpg")
        cv2.imwrite(filename, patch)


for _, n in train_df.iterrows():
    print(os.path.join(TRAIN_DIR, str(n['image_id'])+".jpg"), ":", n['label'])

    if os.path.isdir(os.path.join('/projectnb/cs640grp/students/xthomas/FINAL_PROJECT/patches', str(n['image_id']))):
        print(f"file exists at: {os.path.join(TRAIN_DIR, str(n['image_id'])+'.jpg')}")
    else:
        patches = extract_informative_patches(os.path.join(TRAIN_DIR, str(n['image_id'])+".jpg"))
        save_patches(patches, n['image_id'])
    # plot_patches(patches)    
