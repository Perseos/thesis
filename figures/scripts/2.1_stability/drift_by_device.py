from timedataparser import load_all
import tools, torch
import matplotlib.pyplot as plt

header,time,data,_ = load_all('/home/dgotzens/recording/23-10-13/')
M,K,L = data.shape

rx_id = {int(ramp['RampId']) : int(ramp['RxId']) for chirp in header['Chirps'] for ramp in chirp['Ramps']}
rx_id = [rx_id[k] for k in range(K)]

tx_id = {int(ramp['RampId']) : int(chirp['ChirpId']) for chirp in header['Chirps'] for ramp in chirp['Ramps']}
tx_id = [tx_id[k] for k in range(K)]

startidx = 800

f,ax = plt.subplots(4, sharex=True)
for k in range(K):
    drift = tools.drift(tools.rangedata(data[:,k,startidx:]))
    chip_id = [3,0,1,2][rx_id[k]//4]
    ax[chip_id].plot(time[startidx:], 180/torch.pi * tools.unwrap(drift.angle()))
f.suptitle('Phase Drift Grouped by Rx-Device')
for k in range(4):
    ax[k].grid()
    ax[k].set_xlabel(r'$t$')
    ax[k].set_ylim(-180,45)
    ax[k].set_ylabel(r'$\Delta \phi$ [°]')
    ax[k].set_title(f'chip {k}')
    ax[k].xaxis.set_major_formatter(tools.hhmm)
f.show()


f,ax = plt.subplots(4, sharex=True)
for k in range(K):
    drift = tools.drift(tools.rangedata(data[:,k,startidx:]))
    chip_id = tx_id[k]//3
    ax[chip_id].plot(time[startidx:], 180/torch.pi * tools.unwrap(drift.angle()))
f.suptitle('Phase Drift Grouped by Tx-Device')
for k in range(4):
    ax[k].grid()
    ax[k].set_xlabel(r'$t$')
    ax[k].set_ylim(-180,45)
    ax[k].set_ylabel(r'$\Delta \phi$ [°]')
    ax[k].set_title(f'chip {k}')
    ax[k].xaxis.set_major_formatter(tools.hhmm)
f.show()
input('[Enter]')