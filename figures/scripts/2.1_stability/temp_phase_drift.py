from timedataparser import load_all
import tools, torch
from matplotlib import pyplot as plt

header,time,data,temperature = load_all(f'/home/dgotzens/recording/23-10-09/')
nfft = 2**12
data = tools.rangedata(data, N=nfft)
M,K,L = data.shape

search_idx = torch.tensor([m for m,r in enumerate(tools.ranges(header, nfft)) if 1.4<r<2])
m_refl = tools.reflidx(data.mean(1),search_idx)
drift = torch.empty((K,L), dtype=torch.cfloat)

f,(left,right) = plt.subplots(1,2)
for k in range(K):
    print(f'plotting channel {k}', end='\r')
    drift[k,:] = tools.drift(data[:,k,:], m_refl)
    left.plot(time, 20*drift[k,:].abs().log10(), color='k', linewidth=0.1)
    right.plot(time, 180/torch.pi * tools.unwrap(drift[k,:].angle()), color='k', linewidth=0.1)
left.plot(time, 20 * drift.abs().mean(0).log10(),linewidth=2, label='mean') 
right.plot(time, 180/torch.pi * tools.unwrap(drift.angle()).mean(0),linewidth=2, label='mean') 

left.set_ylabel('drift [dB]')
left.set_xlabel('time [h]')
right.set_ylabel('drift [°]')
right.set_xlabel('time [h]')
left.xaxis.set_major_formatter(tools.hhmm)
right.xaxis.set_major_formatter(tools.hhmm)
left.grid()
right.grid()
left.set_title('Amplitude Drift 23-10-09')
right.set_title('Phase Drift 23-10-09')
left.legend()
right.legend()
f.show()
input()
