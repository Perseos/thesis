import torch

def calc_image_fft(data, calibration, settings):
    # range fft
    W,N = settings['range_window'], settings['range_fftlen']
    range_data = window_fft(data,W,N,0)
    range_data /= calibration[None,:] # apply calibration

    # azimuth fft
    idx,W,N = settings['azm_ula'], settings['azm_window'], settings['azm_fftlen']
    azimuth = window_fft(range_data[:,idx],W,N,1)

    # elevation ffts
    idx,W = settings['elv_ula'], settings['elv_window']
    M,N,L = settings['range_fftlen'], settings['elv_fftlen'], idx.shape[-1]
    elevation = torch.empty((M,N,L), dtype=torch.cfloat)
    for l in range(L):
        elevation[:,:,l] = window_fft(range_data[:,idx[:,l]],W,N,1)

    # estimate elevation gain
    elevation_est = elevation.abs().mean(-1)
    elevation_est /= elevation_est.max()

    # return 3d image
    img_3d = azimuth[:,:,None] * elevation_est[:,None,:]
    return img_3d

def window_fft(x, window, fftlen, dim):
    fft = torch.fft.fft(x.transpose(dim,-1)*window, fftlen)
    fft = torch.fft.fftshift(fft)/window.sum()
    return fft.transpose(dim,-1)