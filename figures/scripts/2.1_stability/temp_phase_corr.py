import timedataparser as tdp
import matplotlib
import matplotlib.pyplot as plt
import pdfdefaults as pdf
import torch
from math import floor, ceil, sqrt

dates = ['23-09-22_0','23-09-22','23-09-26','23-09-27', '23-09-28', '23-10-02', '23-10-04', '23-10-05']
# dates = ['23-10-09']
folders = [f'/home/dgotzens/localstorage/workspace/masterarbeit/recording/{date}/' for date in dates]
organize_by_rx = 1
n_fft = 1024

header = tdp.header(folders[0])
n_samples = header['SamplesPerRamp']
n_ramps = len(header['Chirps']) * len(header['Chirps'][0]['Ramps'])
datas = [tdp.timedata(folder, n_samples, n_ramps) for folder in folders]
timestamps = [tdp.timestamps(folder) for folder in folders]
# timestamps = [[float((t-stamps[0]).total_seconds()/60) for t in stamps] for stamps in timestamps ]
temperatures = [tdp.temperatures(folder) for folder in folders]
temperatures = [temperature[list(temperature.keys())[0]] for temperature in temperatures] # only interested in cpu-thermal
pdf.setup()

def unwrap(phases):
    result = [phases[0]]
    π = 3.14159265359
    for p in phases[1:]:
        p0 = result[-1]
        if abs(p-p0)>π:
            p -= (p-p0)/abs(p-p0) * 2*π
        result += [p]
    return result

def closest(val, search):
    dists = [abs(s-val) for s in search]
    return dists.index(min(dists))

window = torch.hann_window(n_samples)

rows,cols = (floor(sqrt(len(dates))),ceil(len(dates)/floor(sqrt(len(dates)))))
curr = lambda n : 
f, ax = plt.subplots(rows,cols)

for date, data, stamps, temperature in zip(dates, datas, timestamps, temperatures):
    rangedata = torch.fft.fft(window[:,None,None]*data, dim=0, n=n_fft).mean(dim=1)
    m_refl = rangedata.index_select(dim=0, index=torch.arange(n_fft//10))\
                        .mean(dim=-1)\
                        .abs()\
                        .argmax()
    bandpass = torch.arange(m_refl-n_fft//512,m_refl+n_fft//512).long()

    rms = lambda x, dim=0: x.abs().square().mean(dim=dim).sqrt()
    level = 20*torch.log10(rms(rangedata[bandpass,:]))
    phase = unwrap(rangedata[m_refl,:].angle().tolist())
    phase = [p*180/torch.pi for p in phase]

    temptimes = list(temperature.keys())
    tempidx = [closest(stamp, temptimes) for stamp in stamps]
    temps = list(temperature.values())[tempidx]




ax[0].set_title('Mean Level')
ax[0].set_ylabel('dB')
ax[1].set_title('Mean Phase')
# ax[1].set_yticks(range(-90,91,45), map(lambda x: f'{x}°', range(-90,91,45)))
ax[2].set_title('System Temperature ')
ax[2].set_xlabel('t')
ax[2].xaxis.set_major_formatter(hhmm)


for k in range(3):
    ax[k].legend()
    ax[k].grid()
f.set_size_inches(pdf.a4_textwidth, pdf.a4_textwidth*1.2)
f.savefig('../mean_phase_23-10-09.pdf')