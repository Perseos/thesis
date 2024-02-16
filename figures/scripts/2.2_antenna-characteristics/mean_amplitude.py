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


fig, ax = plt.subplots(2, sharex=True, sharey=True)
fig.set_size_inches(pdf.a4_textwidth, 1.3*pdf.a4_textwidth)
for m, measurement in enumerate(('a','d')):
    for n, dist in enumerate((2,8,18,32)):
        with open(f'{folder}{measurement}{dist:02d}_angle.pkl', 'rb') as f:
            angle = pickle.load(f)
        l_deg = [l for l,a in enumerate(angle) if -90 < a*180/pi-90 < 90]
        angle_deg = [a*180/pi - 90 for l,a in enumerate(angle) if l in l_deg]
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
        gain = fft.abs().mean(1)[m_refl, range(L)]
        ax[m].plot(angle_deg, 20*gain.log10(), label=f'{dist}m')
    ax[m].yaxis.set_major_formatter(EngFormatter('dBFS'))
    ax[m].set_ylabel('level')
    ax[m].set_xlabel('rotation')
    ax[m].xaxis.set_major_formatter(EngFormatter('°'))
    ax[m].grid()
    ax[m].legend()
    ax[m].set_title('horizontal' if m==0 else 'vertical')  
fig.savefig('/home/dgotzens/thesis/figures/mean_amp.pdf')