from t2v import t2v
from WavenetModel import WaveNet

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as pl

class Wavenet_t2v(pl.LightningModule):
    def __init__(self, inputsize, timesize, timeemb, kernelsize, config, dilatation_rates, outsize):
        super(Wavenet_t2v, self).__init__()
        self.blocks = nn.ModuleList()

        self.time_emb = t2v(timesize, timeemb)

        self.fc = nn.Sequential(
            nn.Linear(timeemb + inputsize, config[0]),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.wavenet = WaveNet(kernelsize, config, dilatation_rates, outsize)

    def forward(self, x, t):
        t = self.time_emb(t)
        x = torch.cat((x, t), dim=2)
        x = self.fc(x)
        x = self.wavenet(x)
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