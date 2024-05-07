import torch


def calc_image_hybrid(data, settings, pos):
    M,K = data.shape
    PP = pos.shape[:-1]

    gain = settings['channel gain']                     # P1xP2x...Pn x K
    slope = settings['chirp slope']
    f0 = settings['start frequency']
    c0 = settings['lightspeed']
    Ts = settings['sample period']
    N = settings['fftlen']
    maxdist = settings['maxdist']
    
    fft = torch.fft.fft(data, n=N, dim=0)             
    tau = time_of_flight(settings, pos)                 # P1xP2x...Pn x K
    weights = gain * torch.exp(2j*torch.pi*f0*tau)      # P1xP2x...Pn x K
    m_refl = (c0*tau/2 * N/maxdist).round().int()       # P1xP2x...Pn x K


    fft_flat = fft.flatten(0,-2)
    img = torch.zeros(PP, dtype=torch.cfloat)
    for k in range(K):
        m_refl_flat = m_refl.flatten(0,-2)[:,k]
        fft_sel = fft_flat[m_refl_flat,k]
        weights_flat = weights.flatten(0,-2)[:,k]
        img +=  (weights_flat.conj()*fft_sel).unflatten(0,PP)
    
    return img

def time_of_flight(settings, pos):
    x_tx, x_rx = settings['x_tx'], settings['x_rx']
    y_tx, y_rx = settings['y_tx'], settings['y_rx']
    K=len(x_tx)
    c0 = settings['lightspeed']

    txpos = torch.tensor([x_tx,y_tx,[0]*K]).transpose(0,1)
    rxpos = torch.tensor([x_rx,y_rx,[0]*K]).transpose(0,1)
    # K x 3                   
    pos = pos.unsqueeze(-2)
    # PP x 1 x 3   
    r_tx = (txpos-pos).square().sum(-1).sqrt()  
    r_rx = (rxpos-pos).square().sum(-1).sqrt()
    # sum(PPx1x3 - Kx3, -1) = PPxK
    return (r_tx+r_rx)/c0

        