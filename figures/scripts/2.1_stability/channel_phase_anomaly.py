import timedataparser as tdp
import matplotlib.pyplot as plt
import pdfdefaults as pdf
import torch

folder = '/home/dgotzens/recording/23-09-22/'
refl_dist = 1.6
lightspeed = 2.99792458e8

header = tdp.header(folder)
n_samples = header['SamplesPerRamp']
n_ramps = len(header['Chirps']) * len(header['Chirps'][0]['Ramps'])
ramp_slope = (header['RampHighFrequency'] - header['RampLowFrequency']) / header['RampDuration']
max_dist = 0.5 * lightspeed * header['SampleRate'] / ramp_slope
m_refl = refl_dist/max_dist * n_samples
data = tdp.timedata(folder, n_samples, n_ramps)

window = torch.hann_window(n_samples)
rangedata = torch.fft.fft(window[:,None,None]*data, dim=0)

bandpass = torch.zeros_like(rangedata)
bandpass[int(m_refl-1):int(m_refl+1),:,:] = 1

rms = lambda x, dim=0: x.abs().square().mean(dim=dim).sqrt()
level = rms(rangedata*bandpass)
phase = (rangedata*bandpass).angle().mean(dim=0) *180/torch.pi


gain_var = torch.empty((16,12))
phase_var = torch.empty((16,12))
gain_mean = torch.empty((16,12))
phase_mean = torch.empty((16,12))

for chirp in header["Chirps"]:
    for ramp in chirp["Ramps"]:
        k = int(ramp["RampId"])
        rx_id = int(ramp["RxId"])
        tx_id = int(chirp["ChirpId"])
        gain_var[rx_id,tx_id] = level[k,:].var()
        phase_var[rx_id,tx_id] = phase[k,:].var()
        gain_mean[rx_id,tx_id] = level[k,:].mean()
        phase_mean[rx_id,tx_id] = phase[k,:].mean()


pdf.setup()
f, ax = plt.subplots(2,2)
f.set_figwidth(pdf.a4_textwidth)

for k in range(2):
    for l in range(2):
        ax[k,l].set_ylabel('Rx-ID')
        ax[k,l].set_yticks(list(map(int,range(16))))
        ax[k,l].set_xlabel('Tx-ID')
        ax[k,l].set_xticks(list(map(int,range(12))))

im = ax[0,0].imshow(gain_mean, origin='lower')
plt.colorbar(im, ax=ax[0,0], label='mean RMS')
im = ax[0,1].imshow(phase_mean, origin='lower')
plt.colorbar(im, ax=ax[0,1], label='mean phase [°]')
im = ax[1,0].imshow(gain_var, origin='lower')
plt.colorbar(im, ax=ax[1,0], label='RMS variance')
im = ax[1,1].imshow(phase_var, origin='lower')
plt.colorbar(im, ax=ax[1,1], label='phase variance[°]')

f.savefig('../channel_phase_anomaly.pdf', dpi=pdf.dpi, backend='pgf')

folder = '/home/dgotzens/recording/23-09-26/'
refl_dist = 1.6
lightspeed = 2.99792458e8

header = tdp.header(folder)
n_samples = header['SamplesPerRamp']
n_ramps = len(header['Chirps']) * len(header['Chirps'][0]['Ramps'])
ramp_slope = (header['RampHighFrequency'] - header['RampLowFrequency']) / header['RampDuration']
max_dist = 0.5 * lightspeed * header['SampleRate'] / ramp_slope
m_refl = refl_dist/max_dist * n_samples
data = tdp.timedata(folder, n_samples, n_ramps)

window = torch.hann_window(n_samples)
rangedata = torch.fft.fft(window[:,None,None]*data, dim=0)

bandpass = torch.zeros_like(rangedata)
bandpass[int(m_refl-1):int(m_refl+1),:,:] = 1

rms = lambda x, dim=0: x.abs().square().mean(dim=dim).sqrt()
level = rms(rangedata*bandpass)
phase = (rangedata*bandpass).angle().mean(dim=0)*180/torch.pi

gain_mean1 = torch.empty((16,12))
phase_mean1 = torch.empty((16,12))

for chirp in header["Chirps"]:
    for ramp in chirp["Ramps"]:
        k = int(ramp["RampId"])
        rx_id = int(ramp["RxId"])
        tx_id = int(chirp["ChirpId"])
        gain_mean1[rx_id,tx_id] = level[k,:].mean()
        phase_mean1[rx_id,tx_id] = phase[k,:].mean()

f, ax = plt.subplots(2,2)
f.set_figwidth(pdf.a4_textwidth)

for k in range(2):
    for l in range(2):
        ax[k,l].set_ylabel('Rx-ID')
        ax[k,l].set_yticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'])
        ax[k,l].set_xlabel('Tx-ID')
        ax[k,l].set_xticks([0,1,2,3,4,5,6,7,8,9,10,11],['0','1','2','3','4','5','6','7','8','9','10','11'])


im = ax[0,0].imshow(gain_mean, origin='lower')
ax[0,0].set_title('23-09-22')
plt.colorbar(im, ax=ax[0,0], label='mean RMS')
im = ax[0,1].imshow(gain_mean1, origin='lower')
ax[0,1].set_title('23-09-26')
plt.colorbar(im, ax=ax[0,1], label='mean RMS')
im = ax[1,0].imshow(phase_mean, origin='lower')
plt.colorbar(im, ax=ax[1,0], label='mean phase [°]')
im = ax[1,1].imshow(phase_mean1, origin='lower')
plt.colorbar(im, ax=ax[1,1], label='mean phase [°]')

f.savefig('../comparison.pdf', dpi=pdf.dpi, backend='pgf')