import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class TimeseriesOverlapDataset(Dataset):
    def __init__(self, df, seqlen):
        # df is a datafame with columns 'time' datetime and 'value' float
        self.df = df
        self.seqlen = seqlen

        # data augmentation for time
        self.df['year'] = self.df['time'].dt.year
        self.df['month'] = self.df['time'].dt.month
        self.df['weekday'] = self.df['time'].dt.weekday
        self.df['day'] = self.df['time'].dt.day
        self.df['hour'] = self.df['time'].dt.hour
        self.df['minute'] = self.df['time'].dt.minute

        self.scaler = StandardScaler()
        self.df['value'] = self.scaler.fit_transform(self.df['value'].values.reshape(-1, 1))

        print(df.head(5))


    def __len__(self):
        return len(self.df) - self.seqlen * 2

    def __getitem__(self, idx):
        # idx is the index of the first value in the sequence
        # return a tuple of (x, y)
        # x is a sequence of seqlen values
        # y is a sequence of seqlen values
        # t is a sequence of seqlen time information
        x = self.df.iloc[idx:idx+self.seqlen][['value']].values
        t = self.df.iloc[idx:idx+self.seqlen][['year', 'month', 'weekday', 'day', 'hour', 'minute']].values
        y = self.df.iloc[idx+self.seqlen:idx+self.seqlen*2][['value']].values

        x = torch.from_numpy(x).float()
        t = torch.from_numpy(t).float()
        y = torch.from_numpy(y).float()

        return x, t, y

class TimeseriesRollingDataset(Dataset):
    def __init__(self, df, seqlen):
        # df is a datafame with columns 'time' datetime and 'value' float
        self.df = df
        self.seqlen = seqlen

        # data augmentation for time
        self.df['year'] = self.df['time'].dt.year
        self.df['month'] = self.df['time'].dt.month
        self.df['weekday'] = self.df['time'].dt.weekday
        self.df['day'] = self.df['time'].dt.day
        self.df['hour'] = self.df['time'].dt.hour
        self.df['minute'] = self.df['time'].dt.minute

        self.scaler = StandardScaler()
        self.df['value'] = self.scaler.fit_transform(self.df['value'].values.reshape(-1, 1))

        print(df.head(5))

    def __len__(self):
        return len(self.df)//self.seqlen

    def __getitem__(self, idx):
        # idx is the index of the first value in the sequence
        # return a tuple of (x, y)
        # x is a sequence of seqlen values
        # y is a sequence of seqlen values
        # t is a sequence of seqlen time information
        x = self.df.iloc[idx*self.seqlen:(idx+1)*self.seqlen][['value']].values
        t = self.df.iloc[idx*self.seqlen:(idx+1)*self.seqlen][['year', 'month', 'weekday', 'day', 'hour', 'minute']].values
        y = self.df.iloc[idx*self.seqlen:(idx+1)*self.seqlen][['value']].values

        x = torch.from_numpy(x).float()
        t = torch.from_numpy(t).float()
        y = torch.from_numpy(y).float()
        
        return x, t, y
