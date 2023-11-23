from timedataparser import load_all
import tools
import matplotlib.pyplot as plt
import pdfdefaults as pdf
import torch
import sys
from torch.fft import fft, fftshift, ifftshift

date='23-10-09' if len(sys.argv) <= 1 else sys.argv[1]
header,time,data,temperature,reboots = load_all(f'/home/dgotzens/localstorage/workspace/masterarbeit/recording/{date}/', True)
M,K,L = data.shape

nfft = 2**11
data = tools.rangedata(data, N=nfft)
M,K,L = data.shape
search_idx = torch.tensor([m for m,r in enumerate(tools.ranges(header, nfft)) if 1.4<r<2])
m_refl = tools.reflidx(data.mean(1),search_idx)

calib_len_min = 20
weights = torch.zeros(K, dtype=torch.cfloat)
for l in range(calib_len_min):
	weights += data[m_refl[l],:,l] / calib_len_min
weights = weights.unsqueeze(-1)
print(0 in weights)
data /= weights

xpos,ypos =  tools.array_pos(header)
ula = []
for x in range(max(xpos)+1):
	k=xpos.index(x)
	while ypos[k] != 0:
		k=xpos.index(x,k+1)
		print(k)
	ula += [k]

f, ax = plt.subplots(1,5,sharey=True)
r = tools.ranges(header,nfft)
vmin = 0
vmax = 0
power = 0
for n,l in enumerate(range(0,L+1,L//4)):
	window = torch.hann_window(len(ula))
	img = fftshift(fft(ifftshift(window*data[:,ula,l]))).abs()
	if n==0:
		vmin = img.min()
		vmax = img.max()
	power = float(20*img[m_refl[l],43].log10())
	ax[n].imshow(img[100:200,:], vmin=vmin, vmax=vmax, origin='lower')
	ax[n].set_title(str(time[l]) + f'\nreflector {round(power,1)}dB')
	ax[n].set_ylabel('distance [m]')
	ax[n].set_xlabel('azimuth bin')
	ax[n].grid()
	ax[n].set_yticks(range(0,100,30),[round(r[m],2) for m in range(100,200,30)])

f.show()
input()

