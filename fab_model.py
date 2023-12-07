import pandas as pd
import torch
import os
import torchvision.transforms as T
from torch import nn, einsum
from einops import rearrange
import torch.nn.functional as F
from torchvision import transforms
import torch.nn as nn
from PIL import Image
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from torch.utils.data.dataloader import default_collate
import gc
import lightning as L
from torch.optim.lr_scheduler import StepLR
from lightning.pytorch import Trainer, seed_everything
import torch.nn.functional as f
import torchmetrics
from torchmetrics.functional import accuracy
from torch import Tensor
from lightning.pytorch.callbacks import EarlyStopping
from torchmetrics import Precision
from torchmetrics import Recall
from torchmetrics import F1Score
from torchmetrics import ConfusionMatrix
from lightning.pytorch.callbacks import ModelSummary
from tqdm import tqdm
from sklearn.metrics import f1_score

Image.MAX_IMAGE_PIXELS = None 
class CNNTransformer(nn.Module):
    def __init__(self, num_patches, d_model, nhead, num_encoder_layers, dim_feedforward, dropout):
        super(CNNTransformer, self).__init__()
        # ... (other initializations remain the same)
        self.shared_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsample

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsample

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsample

            nn.Conv2d(128, d_model, kernel_size=3, padding=1),  # Final convolution to get to d_model channels
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Adaptive pooling to get a fixed size output
        )
        # Positional embeddings for the patches
        self.positional_embeddings = nn.Parameter(torch.randn(1, num_patches, d_model))
        # Transformer encoder layers
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        # Attention-based pooling layer
        self.attention_pool = nn.Linear(d_model, 1)
        self.fc = nn.Linear(d_model, 5)  # Assuming 10 is the number of classes

    
    def forward(self, batch_of_patches):
        # Process each image's patches separately
        all_image_outputs = []
        for image_patches in batch_of_patches:  # image_patches is a list of patches for one image
            # Process 4 patches at a time to reduce memory usage
            batch_size = 8
            conv_outputs = []
            for i in range(0, len(image_patches), batch_size):
                batch_patches = torch.stack(image_patches[i:i+batch_size])
                conv_output = self.shared_conv(batch_patches)
                conv_output = conv_output.view(batch_patches.size(0), -1)
                conv_outputs.append(conv_output)
                del batch_patches
                del conv_output
                gc.collect()

            # Concatenate the outputs for all patches of this image
            conv_seq = torch.cat(conv_outputs, dim=0)
            conv_seq = conv_seq.unsqueeze(0) 
            del conv_outputs
            # Add positional embeddings
            if conv_seq.size(0) > self.positional_embeddings.size(1):
                raise ValueError("Number of patches exceeds positional embeddings size.")
            conv_seq += self.positional_embeddings[:, :conv_seq.size(0), :]

            # Pass through transformer encoder
            transformer_output = self.transformer_encoder(conv_seq)
            del conv_seq
            gc.collect()
            # Attention pooling
            attn_weights = torch.softmax(self.attention_pool(transformer_output).squeeze(-1), dim=1)
            pooled_output = einsum('bn,bnd->bd', attn_weights, transformer_output)

            all_image_outputs.append(pooled_output)

        # Combine the outputs for all images in the batch
        batch_output = torch.stack(all_image_outputs, dim=0)

        # Apply fully connected layer to the pooled output for each image
        out = self.fc(batch_output)

        return out.squeeze(1)

class FinetuneDataset(Dataset):
    def __init__(self, img_folder, temp_data_path, transform=None):
        """
        Args:
            img_folder (str): Path to the folder with images.
            temp_data_path (str): Path to the temperature dataset (CSV).
            transform (callable, optional): Optional transform to be applied on the image.
        """
        self.img_folder = img_folder
        self.temp_data = pd.read_csv(temp_data_path)
        
        self.image_files = os.listdir(img_folder)
        self.transform = transform
        self.label_to_int = {label: idx for idx, label in enumerate(self.temp_data['label'].unique())}
        self.temp_data['label'] = self.temp_data['label'].map(self.label_to_int)
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.img_folder, self.image_files[idx])
        image = Image.open(img_path)
        scale_factor = 0.6
        new_width = int(image.size[0] * scale_factor)
        new_height = int(image.size[1] * scale_factor)
        
        # Resize and save the image
        image = image.resize((new_width, new_height))
        patch_size = 1024
        label = self.temp_data['label'][idx]
        
        if self.transform:
            tensor_img = self.transform(image)
        patches_per_dim = tensor_img.size(1) // patch_size, tensor_img.size(2) // patch_size

        # Unfold the image tensor into patches of size 256x256
        # The first dimension is for channels (C), it remains the same
        # The second and third dimensions (H and W) are divided into patches
        patches = tensor_img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)

        # The unfold operation gives us a 5D tensor: (C, H_patches, W_patches, H, W)
        # We need to reshape it into a list of 3D tensors of shape (C, H, W) for each patch
        patches_reshaped = patches.contiguous().view(3, -1, patch_size, patch_size).permute(1, 0, 2, 3)
        patches = [patch for patch in patches_reshaped]
        return patches, label

    def get_label_mapping(self):
        return {v: k for k, v in self.label_to_int.items()}

def custom_collate_fn(batch):
    # Separate patches and labels
    patches = [item[0] for item in batch]  # List of lists of patches
    labels = [item[1] for item in batch]   # List of labels

    # Use default_collate for labels as they are likely uniform in size
    labels = default_collate(labels)

    return patches, labels


def get_dataloaders(dataset):
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True,collate_fn=custom_collate_fn, num_workers = 2)
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False,collate_fn=custom_collate_fn, num_workers = 2)
    #dataloader = DataLoader(dataset, batch_size=2, shuffle=True,collate_fn=custom_collate_fn)

    return train_dataloader, val_dataloader#, test_dataloader

def train_epoch(fabric, model, optimizer, train_dataloader, lossT, epoch):
    total = 0
    correct = 0
    all_targets = []
    all_predictions = []
    for batch_idx,(inputs, targets) in enumerate(tqdm(train_dataloader, desc="Training")):
        output = model(inputs)
        loss = lossT(output, targets)
        probabilities = F.softmax(output, dim=1)
        # Get the predicted class by finding the index with the maximum probability
        _, predicted = torch.max(probabilities, dim=1)
        all_predictions.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        fabric.backward(loss)
        #fabric.clip_gradients(model, optimizer, clip_val=0.25)
        optimizer.step()
        optimizer.zero_grad()
        if batch_idx % 16 == 0:
            accuracy = 100 * correct / total
            f1 = f1_score(all_targets, all_predictions, average='weighted')
            fabric.print(f"epoch: {epoch} - iteration: {batch_idx} - loss {loss.item():.4f}")
            fabric.print(f"epoch: {epoch} - iteration: {batch_idx} - acc {accuracy}")
            fabric.print(f"epoch: {epoch} - iteration: {batch_idx} - F1 {f1:.2f}")
    # Calculate accuracy
    accuracy = 100 * correct / total
    fabric.print(f"Epoch {epoch} - Accuracy: {accuracy:.2f}%")

@torch.no_grad()
def validate(fabric, model, val_dataloader, lossT):
    fabric.print("Validating ...")
    model.eval()
    total = 0
    correct = 0
    all_targets = []
    all_predictions = []
    losses = torch.zeros(len(val_dataloader))
    for k, (inputs, targets) in enumerate(tqdm(val_dataloader, desc="Val")):
        output = model(inputs)
        loss = lossT(output, targets)
        probabilities = F.softmax(output, dim=1)
        
        # Get the predicted class by finding the index with the maximum probability
        _, predicted = torch.max(probabilities, dim=1)
        all_predictions.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        losses[k] = loss.item()
    # Calculate accuracy
    accuracy = 100 * correct / total
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    fabric.print(f"Accuracy: {accuracy:.2f}%")
    fabric.print(f" F1 {f1:.2f}")
    out = losses.mean()
    model.train()
    return out


def train(fabric, model, optimizer, train_dataloader, val_dataloader, lossT, max_epochs=5):
    for epoch in range(max_epochs):
        train_epoch(fabric, model, optimizer, train_dataloader, lossT, epoch)
        val_loss = validate(fabric, model, val_dataloader, lossT)
        fabric.print(f"val loss {val_loss.item():.4f}")

def main():
    L.seed_everything(42)

    num_patches = 400 # Number of patches, for example
    d_model = 64  # Size of the feature vector for each patch
    nhead = 4  # Number of heads in the multi-head attention mechanisms
    num_encoder_layers = 2  # Number of sub-encoder-layers in the transformer encoder
    dim_feedforward = 64  # Dimension of the feedforward network model
    dropout = 0.1
    patch_size = 1024

    fabric = L.Fabric(accelerator = 'cpu')

    transform = transforms.Compose([  # Resize images to a consistent shape
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize
    ])

    # Data
    dataset = FinetuneDataset("./train_images_compressed_80",'./train.csv' ,transform)

    train_dataloader, val_dataloader = get_dataloaders(dataset)

    # Model
    model = CNNTransformer(num_patches, d_model, nhead, num_encoder_layers, dim_feedforward, dropout)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    #loss 
    lossT =  nn.CrossEntropyLoss()

    model, optimizer = fabric.setup(model, optimizer)
    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)
    train(fabric, model, optimizer, train_dataloader, val_dataloader, lossT = lossT)
    torch.save(model.state_dict(), 'model_state_dict.pth')

if __name__ == "__main__":
    main()