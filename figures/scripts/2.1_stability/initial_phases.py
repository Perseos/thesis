import timedataparser as tdp
import matplotlib
import matplotlib.pyplot as plt
import pdfdefaults as pdf
import torch

dates = ['23-09-22_0','23-09-22','23-09-26','23-09-27','23-09-28']
folders = [f'/home/dgotzens/localstorage/workspace/masterarbeit/recording/{date}/' for date in dates]

header = tdp.header(folders[0])
n_samples = header['SamplesPerRamp']
n_ramps = len(header['Chirps']) * len(header['Chirps'][0]['Ramps'])

datas = [tdp.timedata(folder, n_samples, n_ramps).select(dim=2,index=0) for folder in folders]
window = torch.hann_window(n_samples)
rangedatas = [torch.fft.fft(window[:,None]*data) for data in datas]

levels = []
phases = []
for r, (date, rangedata) in enumerate(zip(dates,rangedatas)):
    m_refl = rangedata.index_select(dim=0, index=torch.arange(100)).abs().square().mean(dim=1).argmax()
    bandpass = torch.arange(m_refl-1,m_refl+1)
    levels.append((100*rangedata.index_select(dim=0, index=bandpass).abs().mean(dim=0)).log10().tolist())
    phases.append(rangedata.index_select(dim=0, index=bandpass).angle().mean(dim=0).tolist())

f, ax = plt.subplots(12,16,subplot_kw={'projection': 'polar'})

for r, phase in enumerate(phases):
    phases[r] = [p-phase[0] for p in phase]


colors = ['b','g','r','c','k']
for chirp in header['Chirps']:
    for ramp in chirp['Ramps']:
        k = ramp['RampId']
        m = chirp['ChirpId']
        n = ramp['RxId']
        for color,date,level,phase in zip(colors,dates,levels,phases):
            ax[m,n].stem(phase[k],level[k], color, markerfmt=f'{color}.')
        ax[m,n].set_xticklabels([])
        ax[m,n].set_yticklabels([])  
        ax[m,n].set_ylim(0,max(map(max,levels)))

plt.get_current_fig_manager().full_screen_toggle()
f.show()
input()