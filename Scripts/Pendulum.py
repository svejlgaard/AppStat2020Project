import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, contextlib, sys
from datetime import datetime
from Residuals import get_period
from statsmodels.stats.weightstats import DescrStatsW

class Pendulum():

    def __init__(self, namelist, printing = False):
        self.namelist = namelist
        self.printing = printing
    
    def period(self):

        times = dict()

        

        for name in self.namelist:
            t, terr = get_period(name)
            times.update({f'{name}': [t, terr]})
        
        time_df = pd.DataFrame.from_dict(times)
        time_df = time_df.transpose()
        time_df.columns = (['period','error'])
        
        if not self.printing:
            time_weighted = DescrStatsW(time_df['period'], weights=time_df['error'])

            self.T_mu = time_weighted.mean
            self.T_sigma = time_weighted.std
            self.T_mu_err = self.T_sigma / np.sqrt(len(self.namelist))
            print(f'The pendulum period is {self.T_mu} +- {self.T_mu_err}')
        
        else:
            self.T_mu = time_df['period'].values
            self.T_sigma  = time_df['error'].values

            print(f'{self.namelist[0]}s timing is {self.T_mu[0]:.4f} +- {self.T_sigma[0]:.4f}')

        
    def length(self):

        import glob
        filepath = '../Data/Pendulum/*raw.dat.txt'
        files = glob.glob(filepath)

        data_string = pd.read_csv(files[1], sep=" ")
        data_string = data_string.dropna(axis=1)
        data_string.columns = ['Simone', 'Niall', 'Charl']
        data_string = data_string.transpose()
        data_string.columns = ['top','bottom']

        len_string = data_string['top'] - data_string['bottom']
        
        print(f'The length of the string is {np.mean(len_string.values):.2f} +- {np.std(len_string.values/np.sqrt(3)):.2f}')
        
        data_bob = pd.read_csv(files[0], sep=" ")
        data_bob = data_bob.dropna(axis=1)
        data_bob = data_bob.transpose()
        data_bob.index = ['Simone', 'Niall', 'Charl']

        print(f'The pendulum bob height is {np.mean(data_bob.values):.2f} +- {np.std(data_bob.values/np.sqrt(3)):.2f}')
        
        data_bob = data_bob / 10
        data_bob = data_bob / 2

        total_len = (len_string.values + data_bob.values) / 100

        self.L_mu = np.mean(total_len)
        self.L_sigma = np.std(total_len)
        self.L_mu_err = self.L_sigma / np.sqrt(len(self.namelist))

        if self.printing:
            print(f'The length is {self.L_mu} +- {self.L_sigma}')
    
    def get_g(self):
        self.g = self.L_mu * (2*np.pi / self.T_mu)**2
        first = 1/(self.T_mu**4) * self.L_sigma**2
        second = 4*self.L_mu**2 / (self.T_mu**6) * self.T_sigma**2
        
        self.g_sigma = 4*np.pi**2 * np.sqrt(np.abs(first - second))

        print(f'The value of g is {self.g} +- {self.g_sigma} m/s')
        
    

pendulum = Pendulum(['Simone','Niall','Charl'], printing=False)

pendulum.period()
pendulum.length()
pendulum.get_g()



