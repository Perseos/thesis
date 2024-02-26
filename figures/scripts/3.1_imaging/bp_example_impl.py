import torch

def calc_image_bp(data, settings, pos):
    M,K,L = data.shape
    PP = pos.shape

    gain = settings['channel gain']                     # PP x K
    slope = settings['chirp slope']
    f0 = settings['start frequency']
    Ts = settings['sample period']

    # Compute weights
    tau = time_of_flight(settings, pos).unsqueeze(-2)   # PP x 1 x K
    t = Ts*torch.arange(M)[:,None]                      # M x 1

    freq = 2*torch.pi*slope*tau
    phase = 2*torch.pi*f0*tau          
    weights = gain.unsqueeze(-2) * \
        torch.exp(1j*freq*t) * \
        torch.exp(1j*phase)  # PP x M x K

    # Compute 3D Images
    imgs = torch.empty((*PP,L), dtype=torch.cfloat)
    for l in range(L):
        imgs[...,l] = (weights.conj() * data[:,:,l]).mean((-2,-1))
    return imgs

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

def calc_pos(settings):
    if settings['coordinate type'] == 'cartesian':
        x0,x1,y0,y1,z0,z1 = settings['limits']
        X,Y,Z = settings['resolution']
        x = torch.linspace(x0,x1,X)[:,None,None].expand(-1,Y,Z)
        y = torch.linspace(y0,y1,Y)[None,:,None].expand(X,-1,Z)
        z = torch.linspace(z0,z1,Z)[None,None,:].expand(X,Y,-1)
    elif settings['coordinate type'] == 'cylindrical':
        r0,r1,theta0,theta1,z0,z1 = settings['limits']
        R,Theta,Z = settings['resolution']
        r = torch.linspace(r0,r1,R)
        theta = torch.linspace(theta0,theta1,Theta)
        x = (r[:,None,None] * theta.cos()[None,:,None]).expand(-1,-1,Z)
        y = (r[:,None,None] * theta.sin()[None,:,None]).expand(-1,-1,Z)
        z = torch.linspace(z0,z1,Z)[None,None,:].expand(R,Theta,-1)
    elif settings['coordinate type'] == 'spherical':
        r0,r1,theta0,theta1,phi0,phi1 = settings['limits']
        R,Theta,Phi = settings['resolution']
        r = torch.linspace(r0,r1,R)
        theta = torch.linspace(theta0,theta1,Theta)
        phi = torch.linspace(phi0,phi1,Phi)
        x = r[:,None,None] * theta[None,:,None].cos() * phi[None,None,:].sin()
        y = r[:,None,None] * theta[None,:,None].sin() * phi[None,None,:].sin()
        z = (r[:,None,None] * phi[None,:,None].cos()).expand(-1,Theta,-1) 
    
    return torch.stack((x,y,z),-1)

        