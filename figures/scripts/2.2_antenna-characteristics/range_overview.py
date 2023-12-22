import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import torch, pickle, numpy


# setup
nfft = 2_048
pi = 3.141592653589 
maxdist = 50
bins_per_meter = nfft / maxdist


folder = '/home/dgotzens/recording/'


fig, ax = plt.subplots(4,4, sharex=True, sharey=True)
fig.set_size_inches((20,16))
ranges = numpy.arange(nfft)*maxdist/nfft
for m, orientation in enumerate("abcd"):
    for n, dist in enumerate((2,8,18,32)):

        # load measurement
        print(f'loading {folder}{orientation}{dist:02d}_data.pt...')

        data = torch.load(f'{folder}{orientation}{dist:02d}_data.pt')
        M,K,L = data.shape
        with open(f'{folder}{orientation}{dist:02d}_timestamps.pkl', 'rb') as f:
            timestamps = pickle.load(f)
        with open(f'{folder}{orientation}{dist:02d}_angle.pkl', 'rb') as f:
            angle = pickle.load(f)
        if numpy.isnan(angle[0]):
            angle[0] = angle[1]

        print('processing...')
        # calculate mean 
        window = torch.hann_window(M).unsqueeze(-1)
        data_abs_mean = torch.zeros(nfft,L)
        for k in range(K):
            fft = torch.fft.fft(window*data[:,k,:], n=nfft, dim=0)
            data_abs_mean += fft.abs()/K
        # plot data
        im=ax[n, m].pcolormesh(180/pi*numpy.asarray(angle)-90, ranges, 20*data_abs_mean.log10().numpy(), vmin=-5, vmax=50)
        ax[n, m].set_title(f'orientation {orientation.upper()}, reflector at {dist}.0m')
        ax[n, m].xaxis.set_major_formatter(EngFormatter('°'))
        ax[n, m].yaxis.set_major_formatter(EngFormatter('m'))
        ax[n, -1].set_xlabel('Angle')
        ax[0,m].set_ylabel('Range')
        ax[n, m].grid()

cb = fig.colorbar(im,ax=ax.ravel().tolist())
cb.yaxis.set_major_formatter(EngFormatter('dB'))
fig.savefig('/home/dgotzens/Schreibtisch/range_overview.png')