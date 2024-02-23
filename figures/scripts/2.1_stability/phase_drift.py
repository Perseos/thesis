import matplotlib.pyplot as plt
import torch
import sys
sys.path.append('/home/dgotzens/scripts')
from timedataparser import load_all
import tools
import pdfdefaults as pdf

date='23-10-09' if len(sys.argv) <= 1 else sys.argv[1]
header,time,data,temperature,reboots = load_all(f'/home/dgotzens/recording/{date}/', True)
M,K,L = data.shape

#pdf.setup()
f, ax = plt.subplots(3,sharex=True)
# f, ax = plt.subplots(2,sharex=True)
nfft = 2**11
data = tools.rangedata(data, N=nfft)
M,K,L = data.shape
search_idx = torch.tensor([m for m,r in enumerate(tools.ranges(header, nfft)) if 0.7<r<1.3])
m_refl = tools.reflidx(data.mean(1),search_idx)
drift = torch.empty((K,L), dtype=torch.cfloat)
for k in range(K):
    print(f'plotting channel {k}', end='\r')
    drift[k,:] = tools.drift(data[:,k,:], m_refl)
    for r in range(len(reboots)):
        subtime = [t for t in time if reboots[r] < t and (r+1==len(reboots) or t<reboots[r+1])]
        subdrift = drift[k,time.index(min(subtime)):time.index(max(subtime))+1]

        ax[0].plot(subtime, 20 * torch.log10(subdrift.abs()), color='k', linewidth=0.1)
        ax[1].plot(subtime, 180/torch.pi * tools.unwrap(subdrift.angle()), color='k', linewidth=0.1)
        # ax[1].scatter(time[:-1], 180/torch.pi * drift.angle(), color='k')
for r in range(len(reboots)):
    subtime = [t for t in time if reboots[r] < t and (r+1==len(reboots) or t<reboots[r+1])]
    subdrift = drift[:,time.index(min(subtime)):time.index(max(subtime))+1]
    ax[0].plot(subtime, 20 * torch.log10(subdrift.abs().mean(0)),linewidth=2, label='mean')
    ax[1].plot(subtime, 180/torch.pi * tools.unwrap(subdrift.angle()).mean(0),linewidth=2, label='mean') 
print('')
cmap = plt.get_cmap('tab10')
for n,key in enumerate(temperature.keys()):
    stamps = list(temperature[key].keys())
    temps = list(temperature[key].values())
    for r in range(len(reboots)):
        substamps = [t for t in stamps if reboots[r] < t and (r+1==len(reboots) or t<reboots[r+1])]
        subtemps = temps[stamps.index(min(substamps)):stamps.index(max(substamps))+1]
        # sel = [reboots[k-1] < stamp < reboots[k] for stamp in list(stamps)[:-1]]
        # print(list(compress(stamps,sel))[0])
        if r==0:
            ax[2].plot(substamps, [t*1e-3 for t in subtemps], label=key, color=cmap(n))
        else: 
            ax[2].plot(substamps, [t*1e-3 for t in subtemps], color=cmap(n))
ax[0].set_ylabel('Level Drift [dB]')
# ax[0].set_ylim(-3,3)

ax[1].set_ylabel('Phase Drift [°]')
# ax[1].set_ylim(-135,45)

ax[2].set_ylabel('System Temperature [°C]')
ax[2].set_ylim(30,70)
ax[2].legend()
# ax[1].set_xlabel('Time')
# ax[1].xaxis.set_major_formatter(tools.hhmm)
ax[2].set_xlabel('Time')
ax[2].xaxis.set_major_formatter(tools.hhmm)
for k in range(3):
    ax[k].grid()
    #ax[k].set_rasterized(True)
f.set_size_inches(pdf.a4_textwidth, pdf.a4_textwidth*1.2)
print('saving figure...                                    ')
f.savefig(f'/home/dgotzens/thesis/figures/meas_{date}_phase_drift.pdf')
#plt.show()
