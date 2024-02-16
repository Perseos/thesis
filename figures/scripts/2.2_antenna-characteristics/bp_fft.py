import argparse, torch

parser = argparse.ArgumentParser(
    prog='bp_fft.py',
    description='applies fft and range-gate to pytorch tensor')

parser.add_argument('path')
parser.add_argument('startdist', type=float)
parser.add_argument('enddist', type=float)
parser.add_argument('-m', '--maxdist', type=float, dest='maxdist', default=50)
parser.add_argument('-n', '--nfft', type=int, dest='nfft', default=2**18)
parser.add_argument('-d', '--dim', type=int, dest='dim', default=0)

args = parser.parse_args()

print('loading ' + args.path)
data = torch.load(args.path)
M,K,L = data.shape

window = torch.hann_window(M)
m0,m1 = int(args.startdist/args.maxdist*args.nfft), int(args.enddist/args.maxdist*args.nfft)

fft = torch.empty((m1-m0,K,L), dtype=torch.cfloat)
for l in range(L):
    print(f'processing sample {l}/{L}...        ', end='\r')
    fft[:,:,l] = torch.fft.fft(window[:,None]*data[:,:,l], n=args.nfft, dim=args.dim)[m0:m1,:]

print('\nsaving...')
torch.save(fft, args.path.replace('data', 'bp_fft'))
