import numpy as np
from matplotlib import pyplot as plt
import pdfdefaults as pdf

pdf.setup()

t0,t1,t2 = (100,500,600)
power = np.zeros(1000)
power[t0:t1] = 1
power[t2:] = 1

ambient = 20.0
working = 90.0
temperature = np.full(1000, ambient)
t=np.arange(1000)
C = .01

a = ambient + (working-ambient)*(1-np.exp(-C*(t-t0)))
temperature_1 = a[t1]
print(temperature_1)

b = temperature_1 - (temperature_1-ambient)*(1-np.exp(-C*(t-t1)))
temperature_2 = b[t2]
print(temperature_2)

c = temperature_2 + (working-temperature_2)*(1-np.exp(-C*(t-t2)))

temperature[t0:t1] = a[t0:t1]
temperature[t1:t2] = b[t1:t2]
temperature[t2:] = c[t2:]

f,(top,bottom) = plt.subplots(2, sharex=True)
# f.suptitle('Expected Temperature')
f.set_figwidth(pdf.a4_textwidth)

top.plot(temperature)
top.set_ylabel('temperature')
top.set_yticks((ambient,temperature_2,working),(r'$T_{ambient}$', r'$T_2$', r'$T_{working}$'))
top.grid()

bottom.plot(power)
bottom.set_ylabel('power')
bottom.set_yticks((-1,0,1,2),('','off', 'on',''))
bottom.grid()
bottom.set_xlabel('time')
bottom.set_xticks((t0,t1,t2),(r'$t_0$',r'$t_1$',r'$t_2$'))

f.savefig('../temperature.pdf', dpi=pdf.dpi, backend='pgf')