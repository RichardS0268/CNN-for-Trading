import pandas as pd
from tqdm import tqdm
import numpy as np
import os
from zipfile import ZipFile
import warnings
warnings.filterwarnings('ignore')
import datetime
from joblib import Parallel, delayed
from utils import *


SUPPORTED_INDICATORS = ['MA']

def cal_indicators(tabular_df, indicator_name, parameters):
    if indicator_name == "MA":
        assert len(parameters) == 1, f'Wrong parameters num, expected 1, got {len(parameters)}'
        slice_win_size = int(parameters[0])
        MA = tabular_df['close'].rolling(slice_win_size, min_periods=1).mean()
        return MA # pd.Series



def single_symbol_image(tabular_df, image_size, start_date, sample_rate, indicators, show_volume):
    ''' generate Candlelist images
    
    parameters: [
        tabular_df  -> pandas.DataFrame: tabular data,
        image_size  -> tuple: (H, W), size shouble (32, 15), (64, 60)
        start_date  -> int: truncate extra rows after generating images,
        indicators  -> dict: technical indicators added on the image, e.g. {"MA": [20]},
        show_volume -> boolean: show volume bars or not
    ]
    
    Note: A single day's data occupies 3 pixel (width). First rows's dates should be prior to the start date in order to make sure there are enough data to generate image for the start date.
    
    return -> list: each item of the list is [np.array(image_size), binary, binary, binary]. The last two binary (0./1.) are the label of ret5, ret20
    
    '''
    ind_names = []
    if indicators:
        for i in range(len(indicators)//2):
            ind = indicators[i*2].NAME
            ind_names.append(ind)
            params = str(indicators[i*2+1].PARAM).split(' ')
            tabular_df[ind] = cal_indicators(tabular_df, ind, params)
    
    dataset = []
    lookback = image_size[1]//3
    for d in range(lookback-1, len(tabular_df)):
        # random skip some trading dates
        if np.random.rand(1) > sample_rate:
            continue
        # skip dates before start_date
        if tabular_df.iloc[d]['date'] < start_date:
            continue
        
        price_slice = tabular_df[d-(lookback-1):d+1][['open', 'high', 'low', 'close']+ind_names].reset_index(drop=True)
        volume_slice = tabular_df[d-(lookback-1):d+1][['volume']].reset_index(drop=True)

        # number of no transactions days > 0.2*look back days
        if (1.0*(price_slice[['open', 'high', 'low', 'close']].sum(axis=1)/price_slice['open'] == 4)).sum() > lookback//5: 
            continue
        
        # project price into quantile
        price_slice = (price_slice - np.min(price_slice.values))/(np.max(price_slice.values) - np.min(price_slice.values))
        volume_slice = (volume_slice - np.min(volume_slice.values))/(np.max(volume_slice.values) - np.min(volume_slice.values))

        if not show_volume:
            price_slice = price_slice.apply(lambda x: x*(image_size[0]-1)).astype(int)
        else:
            if image_size[0] == 32:
                price_slice = price_slice.apply(lambda x: x*(25-1)+7).astype(int)
                volume_slice = volume_slice.apply(lambda x: x*(6-1)).astype(int)
            else:
                price_slice = price_slice.apply(lambda x: x*(51-1)+13).astype(int)
                volume_slice = volume_slice.apply(lambda x: x*(12-1)).astype(int)
        
        image = np.zeros(image_size)
        for i in range(len(price_slice)):
            # draw candlelist 
            image[price_slice.loc[i]['open'], i*3] = 255.
            image[price_slice.loc[i]['low']:price_slice.loc[i]['high']+1, i*3+1] = 255.
            image[price_slice.loc[i]['close'], i*3+2] = 255.
            # draw indicators
            for ind in ind_names:
                image[price_slice.loc[i][ind], i*3:i*3+2] = 255.
            # draw volume bars
            if show_volume:
                image[:volume_slice.loc[i]['volume'], i*3+1] = 255.
    
        label_ret5 = 1 if np.sign(tabular_df.iloc[d]['ret5']) > 0 else 0
        label_ret20 = 1 if np.sign(tabular_df.iloc[d]['ret20']) > 0 else 0
        
        entry = [image, label_ret5, label_ret20]
        dataset.append(entry)
    return dataset


class ImageDataSet():
    def __init__(self, win_size, start_date, end_date, mode, indicators=[], show_volume=False, parallel_num=-1):
        ## Check whether inputs are valid
        assert isinstance(start_date, int) and isinstance(end_date, int), f'Type Error: start_date & end_date shoule be int'
        assert start_date < end_date, f'start date {start_date} cannnot be later than end date {end_date}'
        assert win_size in [5, 20], f'Wrong look back days: {win_size}'
        assert mode in ['default'], f'Type Error: {mode}'
        assert indicators is None or len(indicators)%2 == 0, 'Config Error, length of indicators should be even'
        if indicators:
            for i in range(len(indicators)//2):
                assert indicators[2*i].NAME in SUPPORTED_INDICATORS, f"Error: Calculation of {indicators[2*i].NAME} is not defined"
        
        ## Attributes of ImageDataSet
        if win_size == 5:
            self.image_size = (32, 15)
            self.extra_dates = datetime.timedelta(days=40)
        else:
            self.image_size = (64, 60)
            self.extra_dates = datetime.timedelta(days=40)
            
        self.start_date = start_date
        self.end_date = end_date 
        self.mode = mode
        self.indicators = indicators
        self.show_volume = show_volume
        self.parallel_num = parallel_num
        
        ## Load data from zipfile
        self.load_data()
        
        # Log info
        if indicators:
            ind_info = [(self.indicators[2*i].NAME, str(self.indicators[2*i+1].PARAM).split(' ')) for i in range(len(self.indicators)//2)]
        else:
            ind_info = []
        print(f"DataSet Initialized\n \t - Mode:         {self.mode.upper()}\n \t - Image Size:   {self.image_size}\n \t - Time Period:  {self.start_date} - {self.end_date}\n \t - Indicators:   {ind_info}\n \t - Volume Shown: {self.show_volume}")
        
        
    @timer('Load Data', '8')
    def load_data(self):
        if 'data' not in os.listdir():
            print('Download Original Tabular Data')
            os.system("mkdir data && cd data && wget 'https://cloud.tsinghua.edu.cn/f/f0bc022b5a084626855f/?dl=1' -O tabularDf.zip")
            
        if 'data' in os.listdir() and 'tabularDf.zip' not in os.listdir('data'):
            print('Download Original Tabular Data')
            os.system("cd data && wget 'https://cloud.tsinghua.edu.cn/f/f0bc022b5a084626855f/?dl=1' -O tabularDf.zip")
        
        with ZipFile('data/tabularDf.zip', 'r') as z:
            f =  z.open('tabularDf.csv')
            tabularDf = pd.read_csv(f, index_col=0)
            f.close()
            z.close()
            
        # add extra rows to make sure image of start date and returns of end date can be calculated
        padding_start_date = int(str(pd.to_datetime(str(self.start_date)) - self.extra_dates).split(' ')[0].replace('-', ''))
        paddint_end_date = int(str(pd.to_datetime(str(self.end_date)) + self.extra_dates).split(' ')[0].replace('-', ''))
        self.df = tabularDf.loc[(tabularDf['date'] > padding_start_date) & (tabularDf['date'] < paddint_end_date)].copy(deep=False)
        tabularDf = [] # clear memory
        
        self.df['ret5'] = np.zeros(self.df.shape[0])
        self.df['ret20'] = np.zeros(self.df.shape[0])
        self.df['ret5'] = (self.df['close'].pct_change(5)*100).shift(-5)
        self.df['ret20'] = (self.df['close'].pct_change(20)*100).shift(-20)
        
        self.df = self.df.loc[self.df['date'] <= self.end_date]
        
        
    def generate_images(self, sample_rate):
        dataset_all = Parallel(n_jobs=self.parallel_num)(delayed(single_symbol_image)(\
                                        g[1], image_size = self.image_size,\
                                           start_date = self.start_date,\
                                          sample_rate = sample_rate,\
                                           indicators = self.indicators,\
                                          show_volume = self.show_volume
                                        ) for g in tqdm(self.df.groupby('code'), desc=f'Generating Images (sample rate: {sample_rate})'))
        
        self.dataset_squeeze = []
        for symbol_data in dataset_all:
            self.dataset_squeeze = self.dataset_squeeze + symbol_data
        dataset_all = [] # clear memory
        
        return self.dataset_squeeze