import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import torch, pickle


# setup
nfft = 2_048
pi = 3.141592653589 
maxdist = 50
lightspeed = 299_792_458 
f_start, f_end = 76_009_996_288.0, 80_289_505_280.0
t_chirp = 0.000_064_890_002_249_740_060
hertz_per_meter = (f_end-f_start) / t_chirp / lightspeed
bins_per_meter = nfft / maxdist


folder = '/home/dgotzens/recording/'


fig, ax = plt.subplots(2,2, sharex=True, sharey=True)
fig.set_size_inches((16,10))
for n, orientation in enumerate("dabc"):
    for dist in (2,8,18,32):

        # load measurement
        print(f'loading measurement (orientation {orientation.upper()}, range {dist}m)...')

        data = torch.load(f'{folder}{orientation}{dist:02d}_data.pt')
        M,K,L = data.shape
        with open(f'{folder}{orientation}{dist:02d}_timestamps.pkl', 'rb') as f:
            timestamps = pickle.load(f)
        with open(f'{folder}{orientation}{dist:02d}_angle.pkl', 'rb') as f:
            angle = pickle.load(f)

        

        # calculate mean 
        window = torch.hann_window(M).unsqueeze(-1)
        window = window / window.sum() * 2**15
        bp = torch.zeros(nfft)
        bp_start = int((dist-0.5)*bins_per_meter)
        bp_len = int(1*bins_per_meter)
        bp[bp_start:bp_start+bp_len] = torch.hann_window(bp_len)
        bp = bp.unsqueeze(-1)

        data_abs_mean = torch.zeros(L)
        for k in range(K):
            fft = torch.fft.fft(window*data[:,k,:], n=nfft, dim=0)
            data_abs_mean += torch.mean(fft.abs()*bp, dim=0)/K

        with open(f'/home/dgotzens/scripts/2.2_antenna-characteristics/{orientation}{dist:02d}_out.pkl', 'wb') as f:
            pickle.dump((angle, data_abs_mean), f)
        # plot data
        ax[n//2, n%2].plot([180/pi*a-90 for a in angle], 20*data_abs_mean.log10(), label=f'r={dist}m')
    ax[n//2, n%2].yaxis.set_major_formatter(EngFormatter(unit='dB'))
    ax[n//2, n%2].xaxis.set_major_formatter(EngFormatter(unit='°'))
    ax[n//2, n%2].set_title(f'Sensor Orientation {orientation.upper()}')
    ax[n//2, n%2].set_xlabel('Angle')
    ax[n//2, n%2].set_ylabel('Level')
    ax[n//2, n%2].legend()
    ax[n//2, n%2].grid()

fig.savefig('/home/dgotzens/Schreibtisch/mean_amp.png')