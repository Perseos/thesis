import pandas as pd
import numpy as np
import json
import torch
import os.path
import pickle as pkl

def load_all(path, loadreboots=False, loadfeparams=False):
    head = header(path)
    n_samples = head['SamplesPerRamp']
    n_ramps = len(head['Chirps']) * len(head['Chirps'][0]['Ramps'])
    data = timedata(path, n_samples, n_ramps)
    time = timestamps(path)
    time = time[:data.shape[-1]]
    temps = temperatures(path)

    result = (head,time,data,temps)
    if loadreboots:
        result += (reboots(path),)
    if loadfeparams:
        result += (feparams(path),)

    return result

def header(path, use_cache=True):
    if use_cache and os.path.isfile(path+'header.pkl'): 
        with open(path+'header.pkl', 'rb') as f:
            return pkl.load(f)

    print('parsing ' + path + 'header.json')
    with open(path+'header.json', 'r') as file:
        header = json.load(file)
        with open(path+'header.pkl', 'wb') as f:
            pkl.dump(header['value0'], f)
        return header['value0']

def timedata(path, M, K, use_cache=True):
    maxsize = 4e9
    if use_cache and os.path.isfile(path+'time_data.pt'):
        data = torch.load(path+'time_data.pt')
        print(f'loaded data with shape {data.shape}')
    else:
        df = pd.read_csv(path + f'ch_000.csv')
        L = df.values.shape[0]
        l_offset = 0
        if(os.path.isfile(path+'crop.json')):
            with open(path+'crop.json') as file:
                crop = json.load(file)
                l_offset = int(crop['start'])
                L = int(crop['end'])-int(crop['start'])

        data = torch.empty((M, K, L), dtype=torch.cfloat)
        print(f'loading data {(M,K,L)}...')

        for k in range(K):
            # print('parsing ' + path + f'ch_{k:0>{3}}.csv', end='\r')
            print('parsing ' + path + f'ch_{k:0>{3}}.csv')
            df = pd.read_csv(path + f'ch_{k:0>{3}}.csv')
            # channels.append([[complex(cc) for cc in c if str(c)] for c in df.values])
            data[:,k,:] = torch.tensor(df.values.astype(np.cfloat).T)[:,l_offset:l_offset+L]

        print('Done')
        torch.save(data, path+'time_data.pt')
    return data

def timestamps(path, use_cache=True):     
    if use_cache and os.path.isfile(path+'timestamps.pkl'): 
        with open(path+'timestamps.pkl', 'rb') as f:
            return pkl.load(f)
    
    print(f'parsing {path}timestamps.csv')
    df = pd.read_csv(path+'timestamps.csv', header=None)
    timestamps = pd.to_datetime(df[0], format='%Y-%m-%d_%H-%M-%S').to_list()
    with open(path+'timestamps.pkl', 'wb') as f:
        pkl.dump(timestamps,f)
        return timestamps

def temperatures(path, use_cache=True):    
    if use_cache and os.path.isfile(path+'temperatures.pkl'): 
        with open(path+'temperatures.pkl', 'rb') as f:
            return pkl.load(f)
    
    print(f'parsing {path}temp.log')
    temperatures = pd.read_csv(path+'temp.log', index_col='date', date_format='%y-%m-%d %H:%M:%S', parse_dates=[0]).to_dict()
    # df['date'] = df['date'].to_datetime(format='%y-%m-%d %H:%M:%S')
    # temperatures = df.to_dict()
    with open(path+'temperatures.pkl', 'wb') as f:
        pkl.dump(temperatures, f)
    return temperatures

def reboots(path, use_cache=True):
    if use_cache and os.path.isfile(path+'reboots.pkl'): 
        with open(path+'reboots.pkl', 'rb') as f:
            return pkl.load(f)
    
    print(f'parsing {path}reboots.log')        
    df = pd.read_csv(path+'reboots.log', header=None)
    reboots = pd.to_datetime(df[0], format='%y-%m-%d %H:%M:%S').to_list()
    with open(path+'reboots.pkl', 'wb') as f:
        pkl.dump(reboots, f)
    return reboots

def feparams(path, use_cache=True):
    if use_cache and os.path.isfile(path+'feparams.pkl'): 
        with open(path+'feparams.pkl', 'rb') as f:
            return pkl.load(f)

    print('parsing ' + path + 'feparams.json')
    with open(path+'feparams.json', 'r') as file:
        feparams = json.load(file)
        with open(path+'feparams.pkl', 'wb') as f:
            pkl.dump(feparams, f)
        return feparams