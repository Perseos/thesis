import torch, sys
sys.path.append('/home/dgotzens/scripts')
import pdfdefaults as pdf
import tools
from timedataparser import load_all
import matplotlib.pyplot as plt

date='23-10-09' if len(sys.argv) <= 1 else sys.argv[1]
header,time,data,temperature = load_all(f'/home/dgotzens/localstorage/workspace/masterarbeit/recording/{date}/')

nfft = 2**21
M,K,L = data.shape
m_to_r = tools.ranges(header, nfft)
search_idx = torch.tensor([m for m,r in enumerate(m_to_r) if 1.4<r<2])
m_refl = [tools.reflidx(tools.rangedata(data[:,:,l].mean(1), nfft),search_idx) for l in range(L)]
del data
print('found idx')
# shift = tools.reflidx(tools.rangedata(data[:,0,:], N=nfft))
# shift -= shift.clone()[0]

# pdf.setup()
f,ax = plt.subplots()
f.set_figwidth(pdf.a4_textwidth)

ax.plot(time, [m_to_r[m] for m in m_refl])
ax.xaxis.set_major_formatter(tools.hhmm)
ax.set_xlabel('time')
ax.set_ylabel('distance [mm]')
ax.grid()
f.savefig('/home/dgotzens/thesis/figures/refldist.pdf')
