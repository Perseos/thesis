import pandas as pd
import numpy as np
import json
import torch

def header(path):
    with open(path+'header.json', 'r') as file:
        header = json.load(file)
        return header['value0']

def time_data(path, n_samples, n_channels):
    result = []
    for i in range(0,n_channels):
        print('reading ' + path + f'ch_{i:0>{3}}.csv', end='\r')
        df = pd.read_csv(path + f'ch_{i:0>{3}}.csv')
        result.append(df.values.astype(complex).tolist())
    print('')
    # shape: channel x ramp x sample -> sample x channel x ramp ([k,l,m]->[m,k,l])
    return torch.tensor(result).permute(2,0,1)
    
