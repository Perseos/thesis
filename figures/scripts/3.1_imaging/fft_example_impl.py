import torch

def calc_image(data, weights, settings):
    # range FFT
    N, window = settings['N_range'], settings['range_window']
    range_fft = torch.fft.fftshift(torch.fft.fft(window[:,None]*data, N, 0),0)
    range_fft /= window.sum()
    # apply calibration
    range_fft /= weights        

    
    if not settings['enable_elevation']:
        N, window = settings['N_azm'], settings['azm_window']
        ula = settings['ula_idx'][:,0]
        azimuth_fft = torch.fft.fftshift(torch.fft.fft(window*range_fft[:,ula],N,1))
        return azimuth_fft
    
    # reshape into ULA
    idx = settings['ula_idx']
    padded_shape = (settings['N_range'], *idx.shape)
    fft_padded = torch.zeros(padded_shape, dtype=torch.cfloat)

    for m in range(idx.shape[0]):
        for n in range(idx.shape[1]):
            if idx[m,n] >= 0:
                fft_padded[:,m,n] = range_fft[:,idx[m,n]]
    
    # azimuth FFT
    N, window = settings['N_azm'], settings['azm_window']
    azimuth_fft = torch.fft.fft(window[None,:,None]*fft_padded,N,1)
    azimuth_fft /= window.sum()

    # deconvolve
    gaps = idx>=0
    gaps_fft = torch.fft.fft(window[:,None]*gaps, N, 0)
    azimuth_fft /= (gaps_fft+1) # +1 to avoid divergence
    # elevation FFT
    N, window = settings['N_elv'], settings['elv_window']
    elevation_fft = torch.fft.fft(window*azimuth_fft, N, 2)
    elevation_fft /= window.sum()
    
    return elevation_fft
