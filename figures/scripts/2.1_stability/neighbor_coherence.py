from timedataparser import load_all
from matplotlib import cm
import pdfdefaults as pdf
import matplotlib.pyplot as plt
import torch, tools, sys

date='23-10-09' if len(sys.argv) <= 1 else sys.argv[1]
header,time,data,temperature,reboots = load_all(f'/home/dgotzens/localstorage/workspace/masterarbeit/recording/{date}/', True)
M,K,L = data.shape

pdf.setup()
# f, ax = plt.subplots(2,sharex=True)
nfft = 2**11
data = tools.rangedata(data, N=nfft)
M,K,L = data.shape
search_idx = torch.tensor([m for m,r in enumerate(tools.ranges(header, nfft)) if 1.4<r<2])
m_refl = tools.reflidx(data.mean(1),search_idx)
x_pos,y_pos = tools.array_pos(header)
X = max(x_pos)+1
ids = [min(\
            [x_pos.index(x, k) \
            for k in range(K) \
            if y_pos[k]==0 and x in x_pos[k:]]\
        ) for x in range(X)]

drift = torch.empty((X,L), dtype=torch.cfloat)
for x,k in enumerate(ids):
    drift[x,:] = tools.drift(data[:,k,:], m_refl)  
    col = [c/256. for c in cm.jet(x/X, bytes=True)]
    if x>0:
        dif = drift[x,:] / drift[x-1,:]
        plt.plot(time, 180/torch.pi*tools.unwrap(dif.angle()), \
                  color=col, label=None if x%10 else f'x={x}', linewidth=0.1)
    # plt.plot(dif[x,:].angle(), color=col, label=None if x%10 else x)
plt.gca().xaxis.set_major_formatter(tools.hhmm)
plt.grid()
plt.xlabel('time')
plt.ylabel('difference to neighbor [°]')
plt.legend()
plt.gcf().set_figwidth(pdf.a4_textwidth)
plt.savefig(f'../meas_{date}_neighbor_coherence.pdf')



    

