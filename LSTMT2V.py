import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from t2v import t2v

import pytorch_lightning as pl

class LSTM_t2v(pl.LightningModule):
    def __init__(self, inputsize, timesize, timeemb, hidden_size, num_layers, dropout, outputsize):
        super(LSTM_t2v, self).__init__()
        self.time_emb = nn.Sequential(
            t2v(timesize, timeemb),
            nn.Dropout(dropout)    
        )

        self.lstm = nn.LSTM(inputsize + timeemb, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, outputsize)

    def forward(self, x, t):
        t = self.time_emb(t)
        x = torch.cat((x, t), dim=2)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

    def general_forward(self, x, t):
        return self(x, t)

    def training_step(self, batch, batch_idx):
        x, t, y = batch
        y_hat = self(x, t)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, on_step = True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, t, y = batch
        y_hat = self(x, t)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, t, y = batch
        y_hat = self(x, t)
        loss = F.mse_loss(y_hat, y)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        op = optim.AdamW(self.parameters(), lr=0.0001)
        
        return op
