from timedataparser import load_all
import matplotlib.pyplot as plt
import pdfdefaults as pdf
import torch, tools
import sys
import sys

date='23-10-09' if len(sys.argv) <= 1 else sys.argv[1]
header,_,data,_ = load_all(f'/home/dgotzens/recording/{date}/')

pdf.setup()
plt.figure().set_figwidth(0.5*pdf.a4_textwidth)
ranges = tools.ranges(header, 2**10)
intensities = 20*tools.rangedata(data, 2**10).abs().mean(dim=(1,-1)).log10()
plt.plot(ranges, intensities)
plt.minorticks_on()
plt.grid(which='both')
plt.ylabel('intensity [dB]')
plt.xlabel('distance [m]')
plt.savefig('../interference.pdf', dpi=pdf.dpi, backend='pgf')

plt.figure().set_figwidth(0.5*pdf.a4_textwidth)
ranges = tools.ranges(header, 2**10)
intensities = 20*tools.rangedata(data, 2**10).abs().mean(dim=(1,-1)).log10()
plt.plot(ranges[:2**7], intensities[:2**7])
plt.minorticks_on()
plt.grid(which='both')
plt.ylabel('intensity [dB]')
plt.xlabel('distance [m]')
plt.savefig('../interference_zoom.pdf', dpi=pdf.dpi, backend='pgf')