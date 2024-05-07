import numpy as np
from matplotlib import pyplot as plt
import pdfdefaults as pdf
import timedataparser as tdp
import tools

temperatures = tdp.temperatures('/home/dgotzens/recording/23-10-09/')

pdf.setup()
f, ax = plt.subplots()
f.set_figwidth(pdf.a4_textwidth)

for key in temperatures.keys():
    stamps, temps = map(list, zip(*sorted(temperatures[key].items())))
    ax.plot(stamps, [t*1e-3 for t in temps], label=key)

ax.xaxis.set_major_formatter(tools.hhmm)
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$\theta$ [°C]')
ax.legend()
ax.grid()

f.set_size_inches(pdf.a4_textwidth, 0.5*pdf.a4_textwidth)
f.savefig('/home/dgotzens/Schreibtisch/actual_temperature.pdf')
# f.show()
# input()