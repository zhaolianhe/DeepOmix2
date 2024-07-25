import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class MultimodalDataset(Dataset):
    def __init__(self, mutation_file, expression_file, slide_pt_files, labels_file, transform=None):
        self.mutation_data = pd.read_csv(mutation_file).values
        self.expression_data = pd.read_csv(expression_file).values
        self.slide_files = pd.read_csv(slide_pt_files).values
        self.labels = pd.read_csv(labels_file).values
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mutation_sample = self.mutation_data[idx]
        expression_sample = self.expression_data[idx]
        slide_sample = torch.load(self.slide_files[idx][0])
        label = self.labels[idx]

        if self.transform:
            mutation_sample = self.transform(mutation_sample)
            expression_sample = self.transform(expression_sample)
            slide_sample = self.transform(slide_sample)

        return {
            'mutation': torch.tensor(mutation_sample, dtype=torch.float),
            'expression': torch.tensor(expression_sample, dtype=torch.float),
            'slide': slide_sample,
            'label': torch.tensor(label, dtype=torch.float)
        }

def get_dataloader(mutation_file, expression_file, slide_pt_files, labels_file, batch_size, shuffle=True, transform=None):
    dataset = MultimodalDataset(mutation_file, expression_file, slide_pt_files, labels_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# Example usage:
# dataloader = get_dataloader("mutation.csv", "expression.csv", "slide_files.pt", "labels.csv", batch_size=32)
