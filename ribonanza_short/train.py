import warnings
warnings.filterwarnings("ignore")

import os
import sys
import random
import pickle
import yaml

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from pathlib import Path
from scipy.interpolate import PchipInterpolator, CubicSpline


class Config:
    DATA_PATH = Path("/scratch/phys/sin/rna-dataset")
    sample_sub = DATA_PATH / "sample_submission.csv"
    test_seq = DATA_PATH / "test_sequences.csv"
    train_labels = DATA_PATH / "train_labels.csv"
    train_sequences = DATA_PATH / "train_sequences.csv"
    validation_labels = DATA_PATH / "validation_labels.csv"
    validation_sequences = DATA_PATH / "validation_sequences.csv"
    model_config_path = "pairwise.yaml"
    pretrained_weights_path = "RibonanzaNet.pt"
    save_weights_final = "final_ribonanza.pt"
    save_weights_name = "running_ribonanza.pt"
    
    max_len = 384
    batch_size = 4
    max_len_filter = 1000000
    min_len_filter = 10
    seed = 94

config = Config()

# Set seed for reproducibility
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(config.seed)
torch.backends.cudnn.enabled = False

train_sequences = pd.read_csv(config.train_sequences)
train_labels = pd.read_csv(config.train_labels)
val_sequences = pd.read_csv(config.validation_sequences)
val_labels = pd.read_csv(config.validation_labels)

train_labels["pdb_id"] = train_labels.ID.str.rsplit('_', n=1, expand=True).iloc[:,0]
val_labels["pdb_id"] = val_labels.ID.str.rsplit('_', n=1, expand=True).iloc[:,0]

def making_data_dict(sequences_df: pd.DataFrame, labels_df: pd.DataFrame):
    grouped = labels_df.groupby("pdb_id")
    data = {}
    sequences, pdb_ids, all_xyz = [], [], []

    for pdb_id in tqdm(sequences_df["target_id"], desc="prepairing data"):
        if pdb_id not in grouped.groups:
            continue

        xyz = grouped.get_group(pdb_id)[["x_1", "y_1", "z_1"]].to_numpy(dtype="float32")

        sequence = list((sequences_df[sequences_df["target_id"]== pdb_id])["sequence"])[0]

        sequences.append(sequence)
        pdb_ids.append(pdb_id)
        all_xyz.append(xyz)
        
    data["pdb_ids"] = pdb_ids
    data["sequences"] = sequences
    data["all_xyz"] = all_xyz
    
    return data


#  Create Data Dictionaries

train_data_dict = making_data_dict(train_sequences, train_labels)
val_data_dict = making_data_dict(val_sequences, val_labels)

def filter_data_by_nan_and_length(data_dict, max_len_filter=9999999, min_len_filter=10):
    """Filter sequences based on NaN ratio and length constraints."""
    valid_indices = []
    max_len_seen = 0
    
    for i, xyz in enumerate(data_dict["all_xyz"]):
        # Track the maximum length
        if len(xyz) > max_len_seen:
            max_len_seen = len(xyz)
        
        nan_ratio = np.isnan(xyz).mean()
        seq_len = len(xyz)
        
        # Keep sequence if it meets criteria
        if (nan_ratio <= 0.1) and (min_len_filter < seq_len < max_len_filter):
            valid_indices.append(i)
    
    
    # Filter all fields based on valid_indices
    filtered_data = {
        "pdb_ids": [data_dict["pdb_ids"][i] for i in valid_indices],
        "sequences": [data_dict["sequences"][i] for i in valid_indices],
        "all_xyz": [data_dict["all_xyz"][i] for i in valid_indices]
    }
    
    return filtered_data


# Filter 
train_data_dict = filter_data_by_nan_and_length(train_data_dict, 
                                                max_len_filter=config.max_len_filter,min_len_filter=config.min_len_filter)
val_data_dict = filter_data_by_nan_and_length(val_data_dict,
                                              max_len_filter=config.max_len_filter,min_len_filter=config.min_len_filter)

class RNA3D_Dataset(Dataset):
    """
    A PyTorch Dataset for 3D RNA structures.
    """
    def __init__(self, data_dict, max_len=384):
        self.data = data_dict
        self.max_len = max_len
        self.nt_to_idx = {nt: i for i, nt in enumerate("ACGU")}

    def __len__(self):
        return len(self.data["sequences"])

    def __getitem__(self, idx):
        sequence = [self.nt_to_idx[nt] for nt in self.data["sequences"][idx]]
        sequence = torch.tensor(sequence, dtype=torch.long)
        xyz = torch.tensor(self.data["all_xyz"][idx], dtype=torch.float32)
        
        # If sequence is longer than max_len, randomly crop
        if len(sequence) > self.max_len:
            crop_start = np.random.randint(len(sequence) - self.max_len)
            crop_end = crop_start + self.max_len
            sequence = sequence[crop_start:crop_end]
            xyz = xyz[crop_start:crop_end]

        return {"sequence": sequence, "xyz": xyz}


def collate_rna_batch(batch):
    sequences = [item["sequence"] for item in batch]
    xyzs = [item["xyz"] for item in batch]

    # Pad variable-length sequences so batch_size > 1 works.
    sequence_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    xyz_padded = pad_sequence(xyzs, batch_first=True, padding_value=float("nan"))

    return {"sequence": sequence_padded, "xyz": xyz_padded}


 # Create Dataset and DataLoaders

train_dataset = RNA3D_Dataset(train_data_dict, max_len=config.max_len)
val_dataset = RNA3D_Dataset(val_data_dict, max_len=config.max_len)

train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    collate_fn=collate_rna_batch,
 )
val_loader = DataLoader(
    val_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    collate_fn=collate_rna_batch,
 )

from Network import RibonanzaNet


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries = entries

    def print(self):
        print(self.entries)


def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        cfg = yaml.safe_load(file)
    return Config(**cfg)



#  Model Definition
class FinetunedRibonanzaNet(RibonanzaNet):
    def __init__(self, config_obj, pretrained=False, dropout=0.1):
        config_obj.dropout = dropout
        super(FinetunedRibonanzaNet, self).__init__(config_obj)

        if pretrained:
            self.load_state_dict(
                torch.load(config.pretrained_weights_path, map_location="cpu")
            )

        self.dropout = nn.Dropout(p=0.0)
        self.xyz_predictor = nn.Linear(256, 3)

    def forward(self, src):
        sequence_features, _ = self.get_embeddings(
            src, torch.ones_like(src).long().to(src.device)
        )
        xyz_pred = self.xyz_predictor(sequence_features)
        return xyz_pred



# Initialize Model
model_cfg = load_config_from_yaml(config.model_config_path)
model = FinetunedRibonanzaNet(model_cfg, pretrained=True).cuda()

def calculate_distance_matrix(X, Y, epsilon=1e-4):
    return ((X[:, None] - Y[None, :])**2 + epsilon).sum(dim=-1).sqrt()


def dRMSD(pred_x, pred_y, gt_x, gt_y, epsilon=1e-4, Z=10, d_clamp=None):
    pred_dm = calculate_distance_matrix(pred_x, pred_y)
    gt_dm = calculate_distance_matrix(gt_x, gt_y)

    mask = ~torch.isnan(gt_dm)
    mask[torch.eye(mask.shape[0], device=mask.device).bool()] = False

    diff_sq = (pred_dm[mask] - gt_dm[mask])**2 + epsilon
    if d_clamp is not None:
        diff_sq = diff_sq.clamp(max=d_clamp**2)

    return diff_sq.sqrt().mean() / Z


def local_dRMSD(pred_x, pred_y, gt_x, gt_y, epsilon=1e-4, Z=10, d_clamp=30):
    pred_dm = calculate_distance_matrix(pred_x, pred_y)
    gt_dm = calculate_distance_matrix(gt_x, gt_y)

    mask = (~torch.isnan(gt_dm)) & (gt_dm < d_clamp)
    mask[torch.eye(mask.shape[0], device=mask.device).bool()] = False

    diff_sq = (pred_dm[mask] - gt_dm[mask])**2 + epsilon
    return diff_sq.sqrt().mean() / Z


def dRMAE(pred_x, pred_y, gt_x, gt_y, epsilon=1e-4, Z=10):
    pred_dm = calculate_distance_matrix(pred_x, pred_y)
    gt_dm = calculate_distance_matrix(gt_x, gt_y)

    mask = ~torch.isnan(gt_dm)
    mask[torch.eye(mask.shape[0], device=mask.device).bool()] = False

    diff = torch.abs(pred_dm[mask] - gt_dm[mask])
    return diff.mean() / Z


def align_svd_mae(input_coords, target_coords, Z=10):
    mask = ~torch.isnan(target_coords.sum(dim=-1))
    input_coords = input_coords[mask]
    target_coords = target_coords[mask]

    centroid_input = input_coords.mean(dim=0, keepdim=True)
    centroid_target = target_coords.mean(dim=0, keepdim=True)

    input_centered = input_coords - centroid_input
    target_centered = target_coords - centroid_target

    cov_matrix = input_centered.T @ target_centered
    U, S, Vt = torch.svd(cov_matrix)
    R = Vt @ U.T

    if torch.det(R) < 0:
        Vt_adj = Vt.clone()
        Vt_adj[-1, :] = -Vt_adj[-1, :]
        R = Vt_adj @ U.T

    aligned_input = (input_centered @ R.T) + centroid_target
    return torch.abs(aligned_input - target_coords).mean() / Z

def train_model(model, train_dl, val_dl, epochs=50, cos_epoch=35, lr=3e-4, clip=1):
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.0, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=(epochs - cos_epoch) * len(train_dl),
    )

    best_val_loss = float("inf")
    best_preds = None

    for epoch in range(epochs):
        model.train()
        train_pbar = tqdm(train_dl, desc=f"Training Epoch {epoch+1}/{epochs}")
        running_loss = 0.0

        for idx, batch in enumerate(train_pbar):
            sequence = batch["sequence"].cuda()
            gt_xyz = batch["xyz"].cuda()

            pred_xyz = model(sequence)

            loss = 0.0
            batch_size = sequence.shape[0]
            for b in range(batch_size):
                loss = loss + dRMAE(pred_xyz[b], pred_xyz[b], gt_xyz[b], gt_xyz[b]) + align_svd_mae(pred_xyz[b], gt_xyz[b])
            loss = loss / batch_size

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            optimizer.zero_grad()

            if (epoch + 1) > cos_epoch:
                scheduler.step()

            running_loss += loss.item()
            avg_loss = running_loss / (idx + 1)
            train_pbar.set_description(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

        model.eval()
        val_loss = 0.0
        val_preds = []

        inds_to_exclude = [3, 4, 6, 8, 11, 12, 13, 14, 18, 21, 22, 23]

        with torch.no_grad():
            for i, batch in enumerate(val_dl):
                if i in inds_to_exclude: continue
                sequence = batch["sequence"].cuda()
                gt_xyz = batch["xyz"].cuda()

                pred_xyz = model(sequence)

                batch_loss = 0.0
                batch_size = sequence.shape[0]
                for b in range(batch_size):
                    batch_loss = batch_loss + dRMAE(pred_xyz[b], pred_xyz[b], gt_xyz[b], gt_xyz[b])
                batch_loss = batch_loss / batch_size

                val_loss += batch_loss.item()

                val_preds.append((gt_xyz.cpu().numpy(), pred_xyz.cpu().numpy()))

        val_loss /= len(val_dl)
        print(f"Validation Loss (Epoch {epoch+1}): {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_preds = val_preds
            torch.save(model.state_dict(), config.save_weights_name)
            print(f"  -> New best model saved at epoch {epoch+1}")
        torch.save(model.state_dict(), f"running_model_{epoch}.pt")

    torch.save(model.state_dict(), config.save_weights_final)
    return best_val_loss, best_preds

best_loss, best_predictions = train_model(
    model=model,
    train_dl=train_loader,
    val_dl=val_loader,
    epochs=20,
    cos_epoch=15,
    lr=3e-4,
    clip=1
)
print(f"Best Validation Loss: {best_loss:.4f}")