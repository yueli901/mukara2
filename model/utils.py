import torch
import random
from config import TRAINING

# Set random seeds for reproducibility
torch.manual_seed(TRAINING['seed'])
random.seed(TRAINING['seed'])

class Scaler:
    def __init__(self, data):
        """
        Standard scaler that normalizes data while ignoring NaN values.
        """
        valid_data = torch.where(torch.isnan(data), torch.zeros_like(data), data)
        count = torch.sum(~torch.isnan(data)).float()

        self.mean = torch.sum(valid_data) / count
        self.std = torch.sqrt(torch.sum((valid_data - self.mean) ** 2) / count)

        print("Mean:", self.mean.item(), "Standard Deviation:", self.std.item())

    def transform(self, data):
        return torch.where(torch.isnan(data), data, (data - self.mean) / (self.std + 1e-8))

    def inverse_transform(self, data):
        return torch.where(torch.isnan(data), data, data * self.std + self.mean)

def MSE_Z(label_z, pred_z, scaler):
    mse_z = torch.mean((label_z - pred_z) ** 2)
    # print(f"GT_Z: {label_z}, Pred_Z: {pred_z}, MSE_Z: {mse_z}")
    return mse_z

def MAE(label_z, pred_z, scaler):
    pred = scaler.inverse_transform(pred_z)
    label = scaler.inverse_transform(label_z)
    mae = torch.mean(torch.abs(label - pred))
    # print(f"GT: {label}, Pred: {pred}, MAE: {mae}")
    return mae

def MSE(label_z, pred_z, scaler):
    pred = scaler.inverse_transform(pred_z)
    label = scaler.inverse_transform(label_z)
    mse = torch.mean((label - pred) ** 2)
    # print(f"GT: {label}, Pred: {pred}, MSE: {mse}")
    return mse

def MGEH(label_z, pred_z, scaler):
    pred = scaler.inverse_transform(pred_z)
    label = scaler.inverse_transform(label_z)
    gehs = torch.sqrt(2 * (pred - label) ** 2 / (pred + label + 1e-8))
    mgeh = torch.mean(gehs)
    # print(f"GT: {label}, Pred: {pred}, GEH: {mgeh}")
    return mgeh

def train_test_sampler(edge_ids, true_probability):
    """
    Splits edge IDs into training and test sets based on probability.
    """
    random.shuffle(edge_ids)
    train_ids = [edge_id for edge_id in edge_ids if random.random() < true_probability]
    test_ids = [edge_id for edge_id in edge_ids if edge_id not in train_ids]
    return train_ids, test_ids
