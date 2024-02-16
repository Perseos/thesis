import sys
sys.path.append('/home/dgotzens/scripts/')
from timedataparser import load_all
from matplotlib import cm
from matplotlib.ticker import FuncFormatter
import pdfdefaults as pdf
import matplotlib.pyplot as plt
import tools, torch

dates=[f'23-09-{k}' for k in ('22_0',26,27)] + [f'23-10-{k:02d}' for k in (4,5,9)]

mean_errors = []
durations = []
D=len(dates)
f,ax = plt.subplots(1,D,sharey=True)
f.set_size_inches(pdf.a4_textwidth, pdf.a4_textwidth*1.5)

vmin=-torch.pi/4
vmax=torch.pi/4
for d,date in enumerate(dates):
    print(date)
    header,time,data,temperature,reboots = load_all(f'/home/dgotzens/shares/messdaten/000_Products/iMCR/2023-17-10_Phasenanalyse/{date}/', True)
    M,K,L = data.shape

    # f, ax = plt.subplots(2,sharex=True)
    nfft = 2**11
    data = tools.rangedata(data, N=nfft)
    M,K,L = data.shape
    search_idx = torch.tensor([m for m,r in enumerate(tools.ranges(header, nfft)) if 1.4<r<2])
    m_refl = tools.reflidx(data.mean(1),search_idx)
    x,y = tools.array_pos(header)
    rx,tx = tools.array_id(header)
    neighbor = [min([i for i in range(K) if i==k or x[i] == x[k]-1 and y[i] == y[k]]) for k in range(K)]
    X = max(x)+1
    img = torch.empty((max(rx)+1, max(tx)+1))
    for k in range(K):
        dif = tools.drift(data[:,k,:], m_refl).mean(-1) / tools.drift(data[:,neighbor[k],:],m_refl).mean(-1)
        img[rx[k],tx[k]] = dif.angle()
    if d==0:
        im = ax[d].imshow(img, origin='lower', vmin=vmin, vmax=vmax, cmap='seismic')
    else:
        ax[d].imshow(img, origin='lower', vmin=vmin, vmax=vmax, cmap='seismic')
    ax[d].set_xlabel('tx_id')
    ax[d].set_title(date)
ax[0].set_ylabel('rx_id')
cb = f.colorbar(im, ax=ax, orientation='horizontal', shrink=0.5,\
                format=FuncFormatter(lambda x, pos: f'{x*180/torch.pi:.0f}°'))
f.show()
input()