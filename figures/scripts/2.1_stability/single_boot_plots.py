import timedataparser as tdp
import matplotlib.pyplot as plt
import pdfdefaults as pdf
import torch

folder = '/home/dgotzens/recording/23-09-22/'
header = tdp.header(folder)
n_samples = header['SamplesPerRamp']
n_ramps = len(header['Chirps']) * len(header['Chirps'][0]['Ramps'])
data = tdp.timedata(folder, n_samples, n_ramps)

pdf.setup()
plt.figure().set_figwidth(pdf.a4_textwidth)

plt.plot(torch.real(data[:,0,0]))
plt.grid()
plt.title('First ramp time data (channel 0)')
plt.ylabel('amplitude')
plt.xlabel(r'time index $n$')
plt.savefig('../first_ramp.pdf', dpi=pdf.dpi, backend='pgf')

corr_mat = torch.empty((16,12), dtype=torch.cfloat)
for chirp in header["Chirps"]:
    for ramp in chirp["Ramps"]:
        k = int(ramp["RampId"])
        rx_id = int(ramp["RxId"])
        tx_id = int(chirp["ChirpId"])
        corr_mat[rx_id,tx_id] = torch.sum(data[:,k,0]*data[:,0,0].conj())

f, (left,right) = plt.subplots(1,2, sharey=True)
f.set_figwidth(pdf.a4_textwidth)

im = left.imshow(corr_mat.abs(), origin='lower')
plt.colorbar(im, ax=left)
left.set_ylabel('Rx-ID')
left.set_yticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'])
left.set_xlabel('Tx-ID')
left.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11],['0','1','2','3','4','5','6','7','8','9','10','11'])
left.set_title('Amplitude')

im = right.imshow(corr_mat.angle()*180/torch.pi, origin='lower')
plt.colorbar(im, ax=right)
right.set_xlabel('Tx-ID')
right.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11],['0','1','2','3','4','5','6','7','8','9','10','11'])
right.set_title('Phase')


f.savefig('../first_ramp_xcorr.pdf', dpi=pdf.dpi, backend='pgf')