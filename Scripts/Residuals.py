import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, contextlib, sys
from datetime import datetime

# Sets the directory to the current directory
os.chdir(sys.path[0])

time_signature = datetime.now().strftime("%m%d-%H%M")

random_state = 27

plt.style.use('seaborn')


filename = '../Data/Pendulum/'

filename += input('Please insert the name of the data file: ')

times = np.loadtxt(filename)


print('WARNING: Always removing the first time in the data file')

times = times[1:]

x = times[:,0]

y = times[:,1]

fit_values = np.polyfit(x,y,1, full=True)

fit_param = fit_values[0]
fit_slope = fit_param[0]
fit_cutoff = fit_param[1]

fit_residuals = y - (x*fit_slope + fit_cutoff)

residual_err = np.std(fit_residuals)


fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 10),
                       gridspec_kw={'height_ratios':[4,1]}, sharex=True)
ax[0].errorbar(x,y, yerr=residual_err,fmt='.', label=f'Data with errors from fit {residual_err:.2f} s')
ax[0].plot(x, x*fit_slope+fit_cutoff,label='Fit')

ax[1].plot(x, fit_residuals, label='Residuals')

fig.legend(loc='best')

name = f'{filename.replace(".dat","")}_plot_with_residuals.pdf'

name = name.split('/')[-1]

full_name = '../Figures/Pendulum/'

full_name += name

fig.savefig(full_name)

print(f'The figure has been saved as {full_name}')