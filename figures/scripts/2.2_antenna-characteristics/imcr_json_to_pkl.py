import json, torch, pickle, sys

M = 1022
N_Rx = 16
N_Tx = 12
K = N_Rx*N_Tx
# FSR = 2**8 FALSCH
FSR = 2**11

L_max = 1000

def sel(D:dict, p:str):
    result = D
    for sub in p.split('.'):
        result = result[sub]
    return result


def save_data(data:torch.Tensor, angle:list, timestamp:list, suffix=None):
    torch.save(data, sys.argv[1]+suffix+'_data.pt')
    with open(sys.argv[1]+suffix+'_angle.pkl', 'wb') as handle:
        pickle.dump(angle, handle)
    with open(sys.argv[1]+suffix+'_timestamps.pkl', 'wb') as handle:
        pickle.dump(timestamp, handle)

data = []
angle = []
timestamp = []
current_angle = float('nan')
framecounter = 0
was_split = False
partcounter = 0

for line in sys.stdin:
    j = json.loads(line)
    if j['name'] == 'angle':
        current_angle = (float(sel(j, 'value.value.angle_rad')))
    elif j['name'] == 'data':
        timestamp.append(float(j['time']))
        angle.append(current_angle)
        if framecounter%25 == 0 and framecounter>50:
            t = timestamp[-1]-timestamp[0]
            print(f'{int(t//60)}:{int(t%60):02d}: angle {round(current_angle*180/3.14159265,1)}°, {round(25/(timestamp[-1]-timestamp[-26]), 2)} FPS        ', end='\r')
        frame = torch.zeros((M,K), dtype=torch.cfloat)
        for c in range(N_Tx):
            chirp = sel(j, 'value.Ramps.data')[c]
            imag = torch.tensor([float(sample['imag'])/FSR for sample in chirp['mSamples']])
            real = torch.tensor([float(sample['real'])/FSR for sample in chirp['mSamples']])
            frame[:,c*N_Rx:(c+1)*N_Rx] = torch.complex(real,imag).reshape((N_Rx,M)).T
        data.append(frame)
        framecounter += 1

    if framecounter > L_max:
        print(f'\nSaving data part {partcounter:02d}...')
        save_data(torch.stack(data,-1), angle, timestamp, f'_{partcounter:02}')
        data = []
        angle = []
        timestamp = []
        framecounter = 0
        was_split = True
        partcounter += 1
    
if not was_split:
    print(f'\nSaving data ...')
    save_data(data,angle,timestamp)
elif framecounter > 0:
    print(f'\nSaving data part {partcounter:02d}...')
    save_data(torch.stack(data,-1), angle, timestamp, f'_{partcounter:02}')


