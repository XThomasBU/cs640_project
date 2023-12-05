import os
import gc
import cv2
import math
import copy
import time
import random
import glob
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp
import torchvision
import joblib
from tqdm import tqdm
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from colorama import Fore, Back, Style
from sklearn.metrics import f1_score
b_ = Fore.BLUE
sr_ = Style.RESET_ALL
import warnings
warnings.filterwarnings("ignore")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


# convnext_base
CONFIG = {
    "seed": 40,
    "epochs": 10,
    "img_size": 224,
    "model_name": "vit_base_patch16_224",
    # "checkpoint_path" : "/kaggle/input/tf-efficientnet/pytorch/tf-efficientnet-b0/1/tf_efficientnet_b0_aa-827b6e33.pth",
    "num_classes": 5,
    "train_batch_size": 32,
    "valid_batch_size": 64,
    "learning_rate": 1e-4,
    "scheduler": 'CosineAnnealingLR',
    "min_lr": 1e-6,
    "T_max": 500*10,
    "weight_decay": 1e-6,
    "fold" : 0,
    "n_fold": 5,
    "n_accumulate": 1,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}


def set_seed(seed=40):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
set_seed(CONFIG['seed'])


ROOT_DIR = '/projectnb/cs640grp/materials/UBC-OCEAN_CS640'
TRAIN_DIR = '/projectnb/cs640grp/materials/UBC-OCEAN_CS640/train_images_compressed_80'
TEST_DIR = '/projectnb/cs640grp/materials/UBC-OCEAN_CS640/test_images_compressed_80'


def get_train_file_path(image_id):
    return f"{TRAIN_DIR}/{image_id}.jpg"


images = sorted(glob.glob(f"{TRAIN_DIR}/*.jpg"))

os.makedirs(f"/projectnb/cs640grp/students/xthomas/FINAL_PROJECT/fold_results", exist_ok=True)

for fold in range(CONFIG['n_fold']):

    os.makedirs(f"/projectnb/cs640grp/students/xthomas/FINAL_PROJECT/fold_results/fold{fold}", exist_ok=True)
    # shuffle images
    random.shuffle(images)

    train_images = images[:int(len(images)*0.8)]
    valid_images = images[int(len(images)*0.8):]



    df = pd.read_csv(f"{ROOT_DIR}/train.csv")
    encoder = LabelEncoder()
    df['label'] = encoder.fit_transform(df['label'])
    df['file_path'] = df['image_id'].apply(get_train_file_path)
    df_train = df[ df["file_path"].isin(train_images) ].reset_index(drop=True)

    from sklearn.utils.class_weight import compute_class_weight
    class_labels = np.unique(df['label'])
    class_weights = compute_class_weight(class_weight ='balanced', classes = class_labels, y=df['label'])

    class_weights_dict = dict(zip(class_labels, class_weights))
    print("Class Weights:", class_weights_dict)



    df_val = df[ df["file_path"].isin(valid_images) ].reset_index(drop=True)


    def get_concatenated_df(df):
        dfs_to_concat = []
        patches_dir = '/projectnb/cs640grp/students/xthomas/FINAL_PROJECT/patches'
        for i in range(len(df)):
            image_id = df.loc[i, "image_id"]
            image_path = df.loc[i, "file_path"]
            patch_paths = glob.glob(f"{patches_dir}/{image_id}/*.jpg")

            # Creating a DataFrame for each image
            patch_df = pd.DataFrame({
                "image_id": [image_id] * len(patch_paths),
                "label": [df.loc[i, "label"]] * len(patch_paths),
                "file_path": patch_paths
            })

            # Append the patch DataFrame to the list
            dfs_to_concat.append(patch_df)

        # Concatenate all DataFrames in the list
        df = pd.concat(dfs_to_concat, ignore_index=True)
        return df


    df_train = get_concatenated_df(df_train)
    df_val = get_concatenated_df(df_val)


    with open("label_encoder.pkl", "wb") as fp:
        joblib.dump(encoder, fp)

    CONFIG['T_max'] = df.shape[0] * (CONFIG["n_fold"]-1) * CONFIG['epochs'] // CONFIG['train_batch_size'] // CONFIG["n_fold"]



    skf = StratifiedKFold(n_splits=CONFIG['n_fold'])

    # # Create a new column 'kfold' with default values
    # df['kfold'] = -1

    # # Iterate through unique image IDs and assign folds
    # for image_id in df['image_id'].unique():
    #     indices = df[df['image_id'] == image_id].index
    #     labels = df.loc[indices, 'label']
        
    #     for fold, (_, val_idx) in enumerate(skf.split(indices, labels)):
    #         df.loc[indices[val_idx], 'kfold'] = fold

    # # Convert 'kfold' column to integers
    # df['kfold'] = df['kfold'].astype(int)

    class UBCDataset(Dataset):
        def __init__(self, df, transforms=None):
            self.df = df
            self.file_names = df['file_path'].values
            self.labels = df['label'].values
            self.transforms = transforms
            
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, index):
            img_path = self.file_names[index]
            img_index = os.path.basename(os.path.dirname(img_path))
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            label = self.labels[index]
            
            if self.transforms:
                img = self.transforms(image=img)["image"]
                
            return {
                'image': img,
                'label': torch.tensor(label, dtype=torch.long),
                'index': img_index
            }

    data_transforms = {
        "train": A.Compose([
            A.Resize(CONFIG['img_size'], CONFIG['img_size']),
            A.ShiftScaleRotate(shift_limit=0.1, 
                            scale_limit=0.15, 
                            rotate_limit=60, 
                            p=0.5),
            A.HueSaturationValue(
                    hue_shift_limit=0.2, 
                    sat_shift_limit=0.2, 
                    val_shift_limit=0.2, 
                    p=0.5
                ),
            A.RandomBrightnessContrast(
                    brightness_limit=(-0.1,0.1), 
                    contrast_limit=(-0.1, 0.1), 
                    p=0.5
                ),
            A.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225], 
                    max_pixel_value=255.0, 
                    p=1.0
                ),
            ToTensorV2()], p=1.),
        
        "valid": A.Compose([
            A.Resize(CONFIG['img_size'], CONFIG['img_size']),
            A.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225], 
                    max_pixel_value=255.0, 
                    p=1.0
                ),
            ToTensorV2()], p=1.)
    }


    class GeM(nn.Module):
        def __init__(self, p=3, eps=1e-6):
            super(GeM, self).__init__()
            self.p = nn.Parameter(torch.ones(1)*p)
            self.eps = eps

        def forward(self, x):
            return self.gem(x, p=self.p, eps=self.eps)
            
        def gem(self, x, p=3, eps=1e-6):
            return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
            
        def __repr__(self):
            return self.__class__.__name__ + \
                    '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                    ', ' + 'eps=' + str(self.eps) + ')'


    class UBCModel(nn.Module):
        def __init__(self, model_name, num_classes, pretrained=True, checkpoint_path=None):
            super(UBCModel, self).__init__()
            self.model_name = model_name
            self.model = timm.create_model(model_name, pretrained=pretrained, checkpoint_path=checkpoint_path)
            try:
                in_features = self.model.head.in_features
            except:
                in_features = self.model.classifier.in_features
            self.model.head = nn.Identity()
            self.pooling = GeM()
            self.linear = nn.Linear(in_features, num_classes)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, images):
            if self.model_name == "vit_base_patch16_224":
                output = self.model(images)
            else:
                output = self.model.forward_features(images)
                output = self.pooling(output).flatten(1)
            output = self.linear(output)
            return output

        
    model = UBCModel(CONFIG['model_name'], CONFIG['num_classes'])
    model.to(CONFIG['device'])


    def criterion(outputs, labels, class_weights_dict):
        class_weights = torch.tensor(list(class_weights_dict.values())).to(CONFIG['device']).float()
        return nn.CrossEntropyLoss()(outputs, labels)


    def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch, class_weights_dict):
        model.train()
        
        dataset_size = 0
        running_loss = 0.0
        running_acc  = 0.0

        results_dict = {}
        actual_label_dict = {}
        
        bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for step, data in bar:
            images = data['image'].to(device, dtype=torch.float)
            labels = data['label'].to(device, dtype=torch.long)
            
            batch_size = images.size(0)
            outputs = model(images)
            loss = criterion(outputs, labels, class_weights_dict)
            loss = loss / CONFIG['n_accumulate']
                
            loss.backward()
        
            if (step + 1) % CONFIG['n_accumulate'] == 0:
                optimizer.step()

                # zero the parameter gradients
                optimizer.zero_grad()

                if scheduler is not None:
                    scheduler.step()
                    
            _, predicted = torch.max(model.softmax(outputs), 1)
            acc = torch.sum( predicted == labels )
            
            running_loss += (loss.item() * batch_size)
            running_acc  += acc.item()
            dataset_size += batch_size
            
            epoch_loss = running_loss / dataset_size
            epoch_acc = running_acc / dataset_size
            
            bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss, Train_Acc=epoch_acc,
                            LR=optimizer.param_groups[0]['lr'])

            for i, idx in enumerate(data['index']):
                if results_dict.get(idx) is None:
                    results_dict[idx] = [predicted.tolist()[i]]
                else:
                    results_dict[idx].append(predicted.tolist()[i])

                if actual_label_dict.get(idx) is None:
                    actual_label_dict[idx] = labels.tolist()[i]
        
        for idx in results_dict.keys():
            results_dict[idx] = max(results_dict[idx], key=results_dict[idx].count)
        
        # get accuracy by comparing results_dict and actual_label_dict
        correct_predictions = 0
        for idx in results_dict.keys():
            if results_dict[idx] == actual_label_dict[idx]:
                correct_predictions += 1
        print('sanity check', len(results_dict), len(actual_label_dict), correct_predictions, len(results_dict) - correct_predictions)
        actual_accuracy = correct_predictions / len(results_dict)


        train_voting_f1 = f1_score(list(actual_label_dict.values()), list(results_dict.values()), average='macro')

        print(f"Train Voting Accuracy: {actual_accuracy * 100:.2f}%")
        print(f"Train Voting F1: {train_voting_f1 :.2f}")
        gc.collect()
        
        return epoch_loss, epoch_acc, actual_accuracy, train_voting_f1


    @torch.inference_mode()
    def valid_one_epoch(model, dataloader, device, epoch, class_weights_dict):
        model.eval()
        
        dataset_size = 0
        running_loss = 0.0
        running_acc = 0.0

        results_dict = {}
        actual_label_dict = {}
        
        bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for step, data in bar:        
            images = data['image'].to(device, dtype=torch.float)
            labels = data['label'].to(device, dtype=torch.long)
            
            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, labels, class_weights_dict)

            _, predicted = torch.max(model.softmax(outputs), 1)
            acc = torch.sum( predicted == labels )

            running_loss += (loss.item() * batch_size)
            running_acc  += acc.item()
            dataset_size += batch_size
            
            epoch_loss = running_loss / dataset_size
            epoch_acc = running_acc / dataset_size
            
            bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss, Valid_Acc=epoch_acc,
                            LR=optimizer.param_groups[0]['lr'])

            for i, idx in enumerate(data['index']):
                if results_dict.get(idx) is None:
                    results_dict[idx] = [predicted.tolist()[i]]
                else:
                    results_dict[idx].append(predicted.tolist()[i])

                if actual_label_dict.get(idx) is None:
                    actual_label_dict[idx] = labels.tolist()[i]
        
        for idx in results_dict.keys():
            results_dict[idx] = max(results_dict[idx], key=results_dict[idx].count)
        
        # get accuracy by comparing results_dict and actual_label_dict
        correct_predictions = 0
        for idx in results_dict.keys():
            if results_dict[idx] == actual_label_dict[idx]:
                correct_predictions += 1
        print('sanity check', len(results_dict), len(actual_label_dict), correct_predictions, len(results_dict) - correct_predictions)
        actual_accuracy = correct_predictions / len(results_dict)

        val_voting_f1 = f1_score(list(actual_label_dict.values()), list(results_dict.values()), average='macro')

        print(f"Validation Voting Accuracy: {actual_accuracy * 100:.2f}%")
        print(f"Validation Voting F1: {val_voting_f1 :.2f}")
        gc.collect()
        
        return epoch_loss, epoch_acc, actual_accuracy, val_voting_f1


    def run_training(model, optimizer, scheduler, device, num_epochs, class_weights_dict, fold):
        if torch.cuda.is_available():
            print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
        
        start = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_epoch_acc = -np.inf
        history = defaultdict(list)
        
        for epoch in range(1, num_epochs + 1): 
            gc.collect()
            train_epoch_loss, train_epoch_acc, actual_accuracy, train_voting_f1 = train_one_epoch(model, optimizer, scheduler, 
                                            dataloader=train_loader, 
                                            device=CONFIG['device'], epoch=epoch, class_weights_dict=class_weights_dict)
            
            val_epoch_loss, val_epoch_acc, val_actual_accuracy, val_voting_f1 = valid_one_epoch(model, valid_loader, device=CONFIG['device'], 
                                            epoch=epoch, class_weights_dict=class_weights_dict)
        
            history['Train Loss'].append(train_epoch_loss)
            history['Valid Loss'].append(val_epoch_loss)
            history['Train Accuracy'].append(train_epoch_acc)
            history['Valid Accuracy'].append(val_epoch_acc)
            history['Valid Voting Accuracy'].append(val_actual_accuracy)
            history['Train Voting Accuracy'].append(actual_accuracy)
            history['Valid Voting F1'].append(val_voting_f1)
            history['Train Voting F1'].append(train_voting_f1)
            history['lr'].append( scheduler.get_lr()[0] )
            
            # deep copy the model
            if best_epoch_acc <= val_actual_accuracy:
                print(f"{b_}Validation Accuracy Improved ({best_epoch_acc} ---> {val_actual_accuracy})")
                best_epoch_acc = val_actual_accuracy
                best_model_wts = copy.deepcopy(model.state_dict())
                PATH = "/projectnb/cs640grp/students/xthomas/FINAL_PROJECT/fold_results/fold{}/Acc{:.2f}_Loss{:.4f}_epoch{:.0f}.bin".format(fold, best_epoch_acc, val_epoch_loss, epoch)
                torch.save(model.state_dict(), PATH)
                # Save a model file from the current directory
                print(f"Model Saved{sr_}")
                
            print()
        
        end = time.time()
        time_elapsed = end - start
        print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
            time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
        print("Best Accuracy: {:.4f}".format(best_epoch_acc))
        
        # load best model weights
        model.load_state_dict(best_model_wts)
        
        return model, history


    def fetch_scheduler(optimizer, num_steps):
        if CONFIG['scheduler'] == 'CosineAnnealingLR':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=CONFIG['T_max'], 
                                                    eta_min=CONFIG['min_lr'])
        elif CONFIG['scheduler'] == 'CosineAnnealingWarmRestarts':
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, num_steps, 
                                                                eta_min=CONFIG['min_lr'])
        elif CONFIG['scheduler'] == None:
            return None
            
        return scheduler


    def prepare_loaders(df_train, df_valid, fold):
        # df_train = df[df.kfold != fold].reset_index(drop=True)
        # df_valid = df[df.kfold == fold].reset_index(drop=True)
        
        train_dataset = UBCDataset(df_train, transforms=data_transforms["train"])
        valid_dataset = UBCDataset(df_valid, transforms=data_transforms["valid"])

        print(f"Train Dataset: {len(train_dataset)}, Validation Dataset: {len(valid_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=CONFIG['train_batch_size'], 
                                num_workers=2, shuffle=True, pin_memory=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['valid_batch_size'], 
                                num_workers=2, shuffle=False, pin_memory=True)

        num_steps = len(train_loader) * CONFIG['epochs']
        
        return train_loader, valid_loader, num_steps


    train_loader, valid_loader, num_steps = prepare_loaders(df_train, df_val, fold=CONFIG["fold"])


    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], 
                        weight_decay=CONFIG['weight_decay'])
    scheduler = fetch_scheduler(optimizer, num_steps)


    model, history = run_training(model, optimizer, scheduler,
                                device=CONFIG['device'],
                                num_epochs=CONFIG['epochs'], class_weights_dict=class_weights_dict, fold=fold)

    history = pd.DataFrame.from_dict(history)
    history.to_csv(f"/projectnb/cs640grp/students/xthomas/FINAL_PROJECT/fold_results/fold{fold}/history.csv", index=False)

    plt.plot( range(history.shape[0]), history["Train Loss"].values, label="Train Loss")
    plt.plot( range(history.shape[0]), history["Valid Loss"].values, label="Valid Loss")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    plt.savefig(f"/projectnb/cs640grp/students/xthomas/FINAL_PROJECT/fold_results/fold{fold}/loss.png")

    # new figure
    plt.figure()
    plt.plot( range(history.shape[0]), history["Train Accuracy"].values, label="Train Accuracy")
    plt.plot( range(history.shape[0]), history["Valid Accuracy"].values, label="Valid Accuracy")
    plt.plot( range(history.shape[0]), history["Valid Voting Accuracy"].values, label="Valid Voting Accuracy")
    plt.plot( range(history.shape[0]), history["Train Voting Accuracy"].values, label="Train Voting Accuracy")
    plt.xlabel("epochs")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.legend()
    plt.savefig(f"/projectnb/cs640grp/students/xthomas/FINAL_PROJECT/fold_results/fold{fold}/accuracy.png")


    plt.figure()
    plt.plot( range(history.shape[0]), history["Valid Voting F1"].values, label="Valid Voting F1")
    plt.plot( range(history.shape[0]), history["Train Voting F1"].values, label="Train Voting F1")
    plt.xlabel("epochs")
    plt.ylabel("F1")
    plt.grid()
    plt.legend()
    plt.savefig(f"/projectnb/cs640grp/students/xthomas/FINAL_PROJECT/fold_results/fold{fold}/f1.png")

