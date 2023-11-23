import timedataparser as tdp
import matplotlib
import matplotlib.pyplot as plt
import pdfdefaults as pdf
import torch
import random

dates = ['23-09-22','23-09-26','23-09-27', '23-09-28']
folders = [f'/home/dgotzens/localstorage/workspace/masterarbeit/recording/{date}/' for date in dates]
organize_by_rx = 1
n_fft = 1024

header = tdp.header(folders[0])
n_samples = header['SamplesPerRamp']
n_ramps = len(header['Chirps']) * len(header['Chirps'][0]['Ramps'])

rx_ids = {int(ramp['RampId']) : int(ramp['RxId']) for chirp in header['Chirps'] for ramp in chirp['Ramps']}
tx_ids = {int(ramp['RampId']) : int(chirp['ChirpId']) for chirp in header['Chirps'] for ramp in chirp['Ramps']}

rx_ids = [rx_ids[k] for k in range(n_ramps)]
tx_ids = [tx_ids[k] for k in range(n_ramps)]

datas = [tdp.timedata(folder, n_samples, n_ramps) for folder in folders]
metadatas = [tdp.metadata(folder) for folder in folders]
pdf.setup()
hhmm = matplotlib.dates.DateFormatter('%H:%M')        

window = torch.hann_window(n_samples)
for date, data, metadata in zip(dates, datas, metadatas):
    f, ax = plt.subplots(3, sharex=True)
    rangedata = torch.fft.fft(window[:,None,None]*data, dim=0, n=n_fft)
    m_refl = rangedata.index_select(dim=0, index=torch.arange(n_fft//10))\
                        .mean(dim=(1,2))\
                        .abs()\
                        .argmax()
    bandpass = torch.arange(m_refl-n_fft//512,m_refl+n_fft//512).long()
    rangedata /= rangedata[m_refl,0,:]
    rangedata /= rangedata[m_refl,:,0].unsqueeze(-1)

    rms = lambda x, dim=0: x.abs().square().mean(dim=dim).sqrt()
    level = 20*torch.log10(rms(rangedata[bandpass,:,:]))
    phase = rangedata[m_refl,:,:].angle()*180.0/torch.pi

    for k in torch.randperm(192):
        colors = ['c','b','k','g']
        chip = rx_ids[k]//4 if organize_by_rx else tx_ids[k]//3
        if organize_by_rx and k in (0,4,8,12) or not any((organize_by_rx, k%48)):
            label = f'chip {chip}'
        else:
            label=None
        ax[0].plot(metadata['radar_ts'][:-1], level[k], colors[chip], label=label, linewidth=0.4)
        ax[1].plot(metadata['radar_ts'][:-1], phase[k], colors[chip], label=label, linewidth=0.4)
    for key in metadata['temps'].keys():
        stamps, temps = map(list, zip(*sorted(metadata['temps'][key].items())))
        ax[2].plot(stamps, [t*1e-3 for t in temps], label=key)

    ax[0].set_ylabel('Level Difference to ch.0 [dB]')
    ax[0].set_ylim(-1,1)
    ax[0].set_title(f'measurement {date}')

    ax[1].set_ylabel('Phase Difference to ch.0 [°]')
    ax[1].set_ylim(-20,20)

    ax[2].set_ylabel('System Temperature [°C]')
    ax[2].set_ylim(30,70)

    ax[2].set_xlabel('Time')
    ax[2].xaxis.set_major_formatter(hhmm)
    for k in range(3):
        ax[k].legend()
        ax[k].grid()
    f.set_size_inches(pdf.a4_textwidth, pdf.a4_textwidth*1.2)
    f.savefig(f'../meas-{date}-phase_difference_by_{"r" if organize_by_rx else "t"}x-chip.pdf')
    # f.show()