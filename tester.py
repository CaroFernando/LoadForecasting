import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader

class ForescastModelTester:
    def __init__(self, model, trainds, valds, testds, num_workers = 1, batch_size=32):
        self.model = model
        self.trainds = trainds
        self.valds = valds
        self.testds = testds
        self.batch_size = batch_size

        self.traindl = DataLoader(self.trainds, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        self.valdl = DataLoader(self.valds, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        self.testdl = DataLoader(self.testds, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

    def train(self, epochs=25):
        trainer = pl.Trainer(gpus=1, max_epochs=epochs)
        trainer.fit(self.model, self.traindl, self.valdl)

    def test(self):
        trainer = pl.Trainer(gpus=1)
        return trainer.test(self.model, self.testdl)

    def predict_test(self):
        preds = []

        for x, t, y in self.testdl:
            #print(x.shape, t.shape, y.shape)
            pred = self.model.general_forward(x, t)
            preds.append(pred.detach().numpy())

        preds = np.concatenate(preds)
        preds = preds.reshape(-1, 1)

        preds = self.testds.scaler.inverse_transform(preds)
        preds = preds.reshape(-1)

        return preds


class ForecastTester:
    def __init__(self, models, trainds, valds, testds, num_workers = 1, epoch_config = None):
        self.models = models
        self.trainds = trainds
        self.valds = valds
        self.testds = testds
        self.epoch_config = epoch_config

        self.testers = []
        for model in self.models:
            self.testers.append(ForescastModelTester(model, self.trainds, self.valds, self.testds, num_workers))

    def train(self, epochs=25):
        if self.epoch_config is None:
            for tester in self.testers:
                tester.train(epochs)
        else:
            for tester, ep in zip(self.testers, self.epoch_config):
                tester.train(ep)

    def test(self):
        results = []
        for tester in self.testers:
            results.append(tester.test())
        return results

    def predict_test(self):
        results = []
        for tester in self.testers:
            results.append(tester.predict_test())
        return results

        