import torch
import matplotlib.dates
from numpy import unwrap as npunwrap

fftlen = 2**14
lightspeed = 299792458

def ranges(header, N=fftlen): 
    slope = (header['RampHighFrequency']-header['RampLowFrequency'])/header['RampDuration']
    maxdist = 0.5 * lightspeed * header['SampleRate'] / slope
    return [0.5*n/N * maxdist for n in range(N)] 

def rangedata(data: torch.Tensor, N=fftlen, dim=0):        
    window = torch.hann_window(data.shape[dim])
    for __ in range(data.dim() - 1):
        window = window.unsqueeze(-1)
        # print(window.shape)
    window.transpose(0,dim)
    # print(data.shape)
    if data.imag.abs().mean() < 1e-7:
        print('converting to real spectrum!')
        return torch.fft.rfft(window*data.real, n=2*N, dim=dim).index_select(dim=dim, index=torch.arange(N))
    else:
        # print(data.imag.abs().mean(), data.imag.abs().mean() < 1e-7)
        return torch.fft.fft(window*data, n=N, dim=dim)

def reflidx(rangedata, search=None, dim=0):
    if search==None:
        search = torch.arange(0, rangedata.shape[0]//10)
    return search.min() + rangedata.index_select(dim=dim, index=search).abs().argmax(dim=dim)

def drift(rangedata, m_refl=None):
    if m_refl==None:
        m_refl=reflidx(rangedata.mean(1))
    result = torch.empty(rangedata.shape[1:], dtype=torch.cfloat)
    
    if m_refl.squeeze().dim() == 0:
        result = rangedata.select(dim=0, index=m_refl) / rangedata.select(dim=0, index=m_refl).select(dim=-1, index=0)
    elif m_refl.squeeze().dim() == 1:
        for l, m in enumerate(m_refl):
            result[...,l] = rangedata[m,...,l]/rangedata[m,...,0]
    elif m_refl.squeeze().dim() == 2:
        for k in range(m_refl.shape[0]):
            for l, in enumerate(m_refl[k,:]):
                result[k,l] = rangedata[m,k,l]/rangedata[m,k,0]
    return result

# how to use:

# rangedata = rangedata(data)
# m_refl = reflidx(rangedata)
# ampdrift_dB = 20*torch.log10(drift(rangedata, m_refl).abs())
# phasedrift_deg = drift(rangedata, m_refl).angle()/torch.pi*180
    
def gettemps(temperatures, times):
    result = []
    temps = list(temperatures.values())
    for time in times:
        difs = [abs(t-time) for t in temperatures.keys()]
        idx = difs.index(min(difs))
        result.append(temps[idx])
    return result

rampslope = lambda header : (header['RampHighFrequency'] -  header['RampLowFrequency']) / header['RampDuration']

def array_id(header):
    K = len(header['Chirps']) * len(header['Chirps'][0]['Ramps'])
    rx_id = {int(ramp['RampId']) : int(ramp['RxId']) for chirp in header['Chirps'] for ramp in chirp['Ramps']}
    tx_id = {int(ramp['RampId']) : int(chirp['ChirpId']) for chirp in header['Chirps'] for ramp in chirp['Ramps']}
    return [rx_id[k] for k in range(K)], [tx_id[k] for k in range(K)]

def array_pos(header):
    K = len(header['Chirps']) * len(header['Chirps'][0]['Ramps'])
    x_pos = {int(ramp['RampId']) : int(ramp['X']) for chirp in header['Chirps'] for ramp in chirp['Ramps']}
    y_pos = {int(ramp['RampId']) : int(ramp['Y']) for chirp in header['Chirps'] for ramp in chirp['Ramps']}
    return [x_pos[k] for k in range(K)], [y_pos[k] for k in range(K)]

def ula_idx(header):
    K = len(header['Chirps']) * len(header['Chirps'][0]['Ramps'])
    xpos,ypos = array_pos(header)

    result = []
    for x in range(max(xpos)+1):
        k = xpos.index(x)
        while ypos[k]!=0:
            k = xpos.index(x,k+1)
        result += [k]
    return result



def deriv(x, dim=-1):
    x0 = torch.zeros_like(x)
    x0[...,1:] = x[...,:-1]
    return x-x0

def runavg(x, n=10, dim=-1):
    result = torch.zeros_like(x)
    T = x.shape[-1]
    for t in range(T):
        result[...,t] = x[...,max(t-n,0):min(t+n,T)].mean(-1)
    return result

unwrap = lambda phase, dim=0, pi=torch.pi: torch.tensor(npunwrap(p=phase,period=2*pi,axis=dim))
hhmm = matplotlib.dates.DateFormatter('%H:%M')

def hist(X, n_bins=None):
    if n_bins==None:
        n_bins=len(X)//10
    xrange = max(X)-min(X)
    xmin = min(X)
    X_norm = [int(n_bins * (x-xmin)/xrange) for x in X]

    return {xmin+xrange*b/n_bins: X_norm.count(b) for b in range(n_bins)}




