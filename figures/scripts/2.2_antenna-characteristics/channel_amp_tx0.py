import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import torch, pickle, sys
sys.path.append('/home/dgotzens/scripts/')
import pdfdefaults as pdf
# setup
pdf.setup()
nfft = 2**10

pi = 3.141592653589 
maxdist = 50
lightspeed = 299_792_458 
f_start, f_end = 76_009_996_288.0, 80_289_505_280.0
t_chirp = 0.000_064_890_002_249_740_060
hertz_per_meter = (f_end-f_start) / t_chirp / lightspeed
bins_per_meter = nfft / maxdist

folder = '/home/dgotzens/scripts/2.2_antenna-characteristics/measured/'
params = pickle.load(open(folder+'feparams.pkl', 'rb'))
tx,ty,rx,ry = params['txPosX'], params['txPosY'], params['rxPosX'], params['rxPosY']


folder = '/home/dgotzens/recording/'

fig,axes = plt.subplots(2,2, sharex=True,sharey=True, layout='compressed')
fig.set_size_inches(pdf.a4_textwidth, 0.6*pdf.a4_textwidth)

measurement = 'a'
k_max = 16 

for ax, dist in zip(axes.flat,(2,8,18,32)):
    with open(f'{folder}{measurement}{dist:02d}_angle.pkl', 'rb') as f:
        angle = pickle.load(f)

    k_sorted = [rx.index(x,0,k_max) for x in sorted(set(rx))]
    l_deg = [angle.index(a) for a in sorted(set(angle)) if 30<=180/pi*a-90<=60]
    angle_deg = [180/pi*angle[l]-90 for l in l_deg]
    data = torch.load(f'{folder}{measurement}{dist:02d}_data.pt')[:,:,l_deg]
    data = data[:,k_sorted,:]
    M,K,L = data.shape
    print(f'loaded data for {dist}m. processing...')

    bp_start = int((dist-0.5)*bins_per_meter)
    bp_len = int(1*bins_per_meter)

    window = torch.hann_window(M)
    window = window / window.sum()

    gain = torch.empty(L)

    fft = torch.fft.fft(window[:,None,None]*data, n=nfft, dim=0)
    m_refl = fft[bp_start:bp_start+bp_len,:,:].abs().mean(1).argmax(0) + bp_start
    gain = fft.abs()[m_refl, :, range(L)].T
    print(gain.shape)
    im = ax.imshow(20*gain.log10()-20*gain.max().log10(), origin='lower', vmin=-30, vmax=0, aspect='auto', cmap='jet')
    ax.set_yticks(range(0,k_max,k_max//4), [f'{round(1000*rx[k_sorted[k]],1)}mm' for k in range(0,k_max,k_max//4)], minor=False)
    ax.set_yticks(range(k_max), minor=True)
    ax.set_ylabel('rx-pos')
    ax.set_xticks(range(0,L,L//3), [f'{round(angle_deg[l],-1)}°' for l in range(0,L,L//3)])
    ax.grid()
    ax.set_title(f'{dist}m')

fig.colorbar(im,ax=axes.ravel(), orientation='horizontal', format=EngFormatter('dBr'), shrink=0.6)
fig.savefig('/home/dgotzens/thesis/figures/channel_amp_tx0.pdf')