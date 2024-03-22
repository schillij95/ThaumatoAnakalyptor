import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl

class LitModel(pl.LightningModule):
    def __init__(self):
        print("instantiating model")
        super().__init__()
        self.layer_1 = torch.nn.Linear(28 * 28, 1024)
        self.layer_2 = torch.nn.Linear(1024, 4096)
        self.layer_3 = torch.nn.Linear(4096, 16384)
        self.layer_4 = torch.nn.Linear(16384, 65536)
        self.layer_5 = torch.nn.Linear(65536, 16384)
        self.layer_6 = torch.nn.Linear(16384, 4096)
        self.layer_7 = torch.nn.Linear(4096, 1024)
        self.layer_8 = torch.nn.Linear(1024, 10)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.layer_1(x))
        x = self.dropout(x)
        x = torch.relu(self.layer_2(x))
        if x.shape[0] % 2 == 0:
            x = self.layer_3(x)
        else:
            x = self.layer_4(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = torch.relu(self.layer_5(x))
        x = self.dropout(x)
        x = torch.relu(self.layer_6(x))
        x = torch.relu(self.layer_7(x))
        x = self.layer_8(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def train_dataloader(self):
        # Create a random dataset
        num_samples = 55000
        input_size = 28 * 28
        num_classes = 10

        X = torch.randn(num_samples, 1, 28, 28)
        y = torch.randint(0, num_classes, (num_samples,))
        
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=64)

if __name__ == '__main__':
    model = LitModel()
    trainer = pl.Trainer(accelerator='gpu', devices=8, strategy='ddp')
    trainer.fit(model)
