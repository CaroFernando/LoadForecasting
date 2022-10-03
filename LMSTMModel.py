import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as pl

class LSTMModel(pl.LightningModule):
    def __init__(self, inputsize, outputsize, hidden_size, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(inputsize, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, outputsize)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

    def general_forward(self, x, t):
        return self(x)

    def training_step(self, batch, batch_idx):
        x, _, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, on_step = True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, _, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        op = optim.AdamW(self.parameters(), lr=0.0001)
        
        return op
