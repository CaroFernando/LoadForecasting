import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

if __name__ == "__main__":
    df = pd.read_csv('time_series_30min_singleindex.csv')
    df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
    print(df.columns)
    time_load_df = df[['utc_timestamp', 'GB_NIR_load_actual_entsoe_transparency']]

    time_load_df = time_load_df.dropna()
    time_load_df.reset_index(drop=True, inplace=True)

    time_load_df.columns = ['time', 'value']

    from TimeseriesDatasets import TimeseriesOverlapDataset, TimeseriesRollingDataset

    # split into train for 2015-2018, val for 2019, test for 2020

    train_df = time_load_df.loc[time_load_df['time'] < '2019-01-01']
    val_df = time_load_df.loc[(time_load_df['time'] >= '2019-01-01') & (time_load_df['time'] < '2020-01-01')]
    test_df = time_load_df.loc[time_load_df['time'] >= '2020-01-01']

    trainds = TimeseriesRollingDataset(train_df, 24*7*2)
    valds = TimeseriesRollingDataset(val_df, 24*7*2)
    testds = TimeseriesRollingDataset(test_df, 24*7*2)

    print(len(trainds), len(valds), len(testds))

    from WavenetT2V import Wavenet_t2v

    wavenet = Wavenet_t2v(6, 31, 3, config = [32, 63, 64, 128, 64], dilatation_rates = [0, 12, 8, 4, 2], outsize = 1)

    from tester import ForecastTester

    tester = ForecastTester([wavenet], trainds, valds, testds)

    tester.train()
