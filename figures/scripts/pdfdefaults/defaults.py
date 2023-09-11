import matplotlib

def setup():
    matplotlib.use('pgf')
    matplotlib.rcParams.update({
    "pgf.texsystem" : "pdflatex",
    "font.family"   : "serif",
    "text.usetex"   : True,
    "pgf.rcfonts"   : False
})
    
a4_textwidth = 6.3349
dpi = 600