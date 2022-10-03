import pandas as pd
import numpy as np

from tester import ForecastTester
from WavenetT2V import Wavenet_t2v
from WavenetModel import WaveNet
from LMSTMModel import LSTMModel
from LSTMT2V import LSTM_t2v
from TimeseriesDatasets import TimeseriesOverlapDataset, TimeseriesRollingDataset

if __name__ == "__main__":
    seqlens = [2*24, 2*24*7, 2*24*30]

    for seqlen in seqlens:

        print("Seqlen: ", seqlen)

        df = pd.read_csv('time_series_30min_singleindex.csv')
        df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
        time_load_df = df[['utc_timestamp', 'IE_load_actual_entsoe_transparency']]


        time_load_df = time_load_df.dropna()
        time_load_df.reset_index(drop=True, inplace=True)

        time_load_df.columns = ['time', 'value']

        train_df = time_load_df[time_load_df['time'] < '2018-01-01']
        val_df = time_load_df[(time_load_df['time'] >= '2018-01-01') & (time_load_df['time'] < '2018-07-01')]
        test_df = time_load_df[(time_load_df['time'] >= '2018-07-01') & (time_load_df['time'] < '2020-01-01')]

        trainds = TimeseriesOverlapDataset(train_df, seqlen, stride = 10)
        valds = TimeseriesOverlapDataset(val_df, seqlen, stride = 10)
        testds = TimeseriesRollingDataset(test_df, seqlen)

        epochs = [25, 28, 25, 20]

        wavenet = WaveNet(3, config=[1, 8, 16, 32, 16], dilatation_rates=[0, 12, 8, 4, 2], outsize=1)
        wavenett2v = Wavenet_t2v(1, 6, 31, 5, config = [64, 128, 256, 256, 128], dilatation_rates = [0, 12, 8, 4, 2], outsize = 1)
        lstm = LSTMModel(1, 1, 128, 2, 0.3)
        lstmt2v = LSTM_t2v(1, 6, 31, 128, 2, 0.3, 1)

        names = ['wavenet', 'wavenett2v', 'lstm', 'lstmt2v']

        tester = ForecastTester([wavenet, wavenett2v, lstm, lstmt2v], trainds, valds, testds, 4, epochs)

        tester.train()

        preds = tester.predict_test()

        predsdf = pd.DataFrame()

        predsdf['time'] = test_df['time'].iloc[seqlen : seqlen + preds[0].shape[0]]
        values = test_df['value'].iloc[seqlen : seqlen + preds[0].shape[0]]
        values = testds.scaler.inverse_transform(values.values.reshape(-1, 1))
        predsdf['value'] = values.reshape(-1)

        for i in range(len(preds)):
            predsdf[names[i]] = preds[i]

        predsdf.to_csv('preds_' + str(seqlen) + '.csv', index=False)
