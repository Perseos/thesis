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


folder = '/home/dgotzens/recording/'

fig,ax = plt.subplots(2,4, sharex=True,sharey=True, layout='compressed')
fig.set_size_inches(pdf.a4_textwidth, 0.6*pdf.a4_textwidth)

for m, measurement in enumerate(('a','d')):
    for n, dist in enumerate((2,8,18,32)):
        with open(f'{folder}{measurement}{dist:02d}_angle.pkl', 'rb') as f:
            angle = pickle.load(f)
        angle_deg = [round(a*180/pi-90) for a in angle]
        l_deg = [angle_deg.index(a) for a in sorted(set(angle_deg))]
        angle_deg = list(sorted(set(angle_deg)))
        data = torch.load(f'{folder}{measurement}{dist:02d}_data.pt')[:,:,l_deg]
        M,K,L = data.shape

        print(f'loaded data for {dist}m. processing...')

        bp_start = int((dist-0.5)*bins_per_meter)
        bp_len = int(1*bins_per_meter)

        window = torch.hann_window(M)
        window = window / window.sum()

        gain = torch.empty(L)

        fft = torch.fft.fft(window[:,None,None]*data, n=nfft, dim=0)
        m_refl = fft[bp_start:bp_start+bp_len,:,:].abs().mean(1).argmax(0) + bp_start
        gain = fft.abs()[m_refl, :, range(L)]
        print(gain.shape)
        im = ax[m,n].imshow(20*gain.T.log10()-20*gain.T.max().log10(), origin='lower', vmin=-30, vmax=0, cmap='jet')
        ax[m,n].set_yticks(range(0,192,16*3), minor=False)
        ax[m,n].set_yticks(range(0,192,16), minor=True)
        ax[m,n].set_xticks(range(0,L,L//6), [f'{round(angle_deg[l],-1)}°' for l in range(0,L,L//6)])
        ax[m,n].grid()
        ax[m,n].set_title(f'{dist}m, ' + ('horizontal','vertical')[m])
    ax[0,n].set_ylabel('channel')


fig.colorbar(im,ax=ax.ravel(), orientation='horizontal', format=EngFormatter('dBr'), shrink=0.6)
fig.savefig('/home/dgotzens/thesis/figures/channel_amp.pdf')