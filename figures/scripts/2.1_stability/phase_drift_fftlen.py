import timedataparser as tdp
import tools
import matplotlib.pyplot as plt
import pdfdefaults as pdf
import torch

dates = ['23-10-09']
folders = [f'/home/dgotzens/localstorage/workspace/masterarbeit/recording/{date}/' for date in dates]

header = tdp.header(folders[0])
n_samples = header['SamplesPerRamp']
n_ramps = len(header['Chirps']) * len(header['Chirps'][0]['Ramps'])

datas = [tdp.timedata(folder, n_samples, n_ramps) for folder in folders]
timestampss = [tdp.timestamps(folder) for folder in folders]
temperaturess = [tdp.temperatures(folder) for folder in folders]
rebootss = [tdp.reboots(folder) for folder in folders]


# pdf.setup()
for date, data, timestamps, temperatures, reboots\
        in zip(dates, datas, timestampss, temperaturess, rebootss):
    f, ax = plt.subplots(4,sharex=True)

    for k in range(data.shape[1]):
        print(f'plotting channel {k}', end='\r')
        for n in range(4):
            drift = tools.drift(tools.rangedata(data[:,k,:], N=2**(6+2*n)))
            ax[n].plot(timestamps[:-1], 180/torch.pi * tools.unwrap(drift.angle()))
            ax[n].set_title(f'FFT_len={2**(6+2*n)}')

    ax[3].set_xlabel('Time')
    ax[3].xaxis.set_major_formatter(tools.hhmm)
    for k in range(4):
        ax[k].grid()
        ax[k].set_ylabel('Phase Drift [°]')
        # ax[k].set_rasterized(True)
    # f.set_size_inches(pdf.a4_textwidth, pdf.a4_textwidth*1.2)
    # f.savefig(f'../meas_{date}_phase_drift.pdf')
    f.show()
    input('\n[enter]')