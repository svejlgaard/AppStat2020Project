import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, contextlib, sys
from datetime import datetime
from Residuals import get_period
from statsmodels.stats.weightstats import DescrStatsW
from scipy.stats import chi2_contingency, chi2
from itertools import combinations

class Pendulum():

    def __init__(self, namelist, printing = False):
        self.namelist = namelist
        self.printing = printing

    
    def get_chi2(self, name, measurements, err):
        chi2_sum = [(i - j)**2 / err[np.argwhere(measurements == i)]**2 for i,j in combinations(measurements,2)]
        true_chi2 = np.sum(chi2_sum)
        true_p = chi2.sf(true_chi2, len(measurements)-1)
        print(f'For the {name} measurements, the Chi2 is {true_chi2:.3f} and p is {true_p:.3f}')

    
    def period(self):
        
        # Create a dictionary for the periods
        times = dict()
        
        # For all the names in the given name list...
        for name in self.namelist:
            # get the period via the get_period function imported from the Residuals script
            t, terr = get_period(name)
            # and save them in a dictionary
            times.update({f'{name}': [t, terr]})
        
        # Transform the dictionary to a pandas dataframe, which makes it easier to perform the 
        # same operations on all imported periods
        time_df = pd.DataFrame.from_dict(times)
        time_df = time_df.transpose()
        time_df.columns = (['period','error'])
        
        self.get_chi2('period', time_df['period'].values, time_df['error'].values)

        if not self.printing:
            # Creating a DescrStatsW class with the data to easily calculate the weighted mean
            # and the weighted errors
            time_weighted = DescrStatsW(time_df['period'], weights=1/(time_df['error'])**2)

            self.T_mu = time_weighted.mean
            self.T_sigma = time_weighted.std
            self.T_mu_err = self.T_sigma / np.sqrt(len(self.namelist))
            print(f'The pendulum period is {self.T_mu} +- {self.T_mu_err}')
        
        else:
            # Only to be used when entering a single name instead of a name list!
            # Will print the period for the given name
            self.T_mu = time_df['period'].values
            self.T_sigma  = time_df['error'].values

            print(f'{self.namelist[0]}s timing is {self.T_mu[0]:.4f} +- {self.T_sigma[0]:.4f}')

        
    def length(self):
        
        import glob
        # Importing all files in the Data/Pendulum directory ending with the well-known file format 'raw.dat.txt'
        filepath = '../Data/Pendulum/*raw.dat.txt'
        files = glob.glob(filepath)

        # The 2nd element in the files list is the string_raw.dat.txt. 
        data_string = pd.read_csv(files[1], sep=" ")

        # Because it's not a csv file, some extra axes are added, but this line removes them
        data_string = data_string.dropna(axis=1)

        # Entering the namelist manually so that this part of the code can also be used,
        # when only looking at data from one of us
        data_string.columns = ['Simone', 'Niall', 'Charl']

        data_string = data_string.transpose()
        data_string.columns = ['top','bottom']
        data_string['error'] = [0.1, 0.1, 0.1]

        # Defining the length of the string 
        len_string = data_string['top'] - data_string['bottom']
        
        err_string = data_string['error']
        self.get_chi2('string',len_string, err_string)


        string_gen = DescrStatsW(len_string, weights=1/(err_string)**2)

        string_mean = string_gen.mean
        string_sigma = string_gen.std
        string_mean_err = string_gen.std / np.sqrt(3)
        
        # Printing the mean and the error on the mean
        print(f'The length of the string is {string_mean:.2f} +- {string_mean_err:.2f}')
        

        # Similar for the pendulum bob...
        data_bob = pd.read_csv(files[0], sep=" ")
        data_bob = data_bob.dropna(axis=1)
        data_bob = data_bob.transpose()
        data_bob.index = ['Simone', 'Niall', 'Charl']
        data_bob.columns = ['height']
        data_bob['error'] = [0.05, 0.05, 0.05]

        bob_weighted = DescrStatsW(data_bob['height'].values, weights=1/(data_bob['error'].values)**2)
        

        self.get_chi2('bob', data_bob['height'].values, data_bob['error'].values)
        

        bob_mean = bob_weighted.mean
        bob_sigma = bob_weighted.std
        bob_mean_err = bob_sigma / np.sqrt(len(self.namelist))

        print(f'The pendulum bob height is {bob_mean:.2f} +- {bob_mean_err:.2f}')
        
        # Dividing by 10 to convert from mm to cm
        data_bob = data_bob / 10

        # Dividing by 2 to get the distance to the centre of mass
        data_bob = data_bob / 2
        # ^^ (Could have been written as a one-liner, but no, it has to be difficuelt)

        # Defining the total length of the pendulum and converting from cm to m
        total_len = (len_string.values + data_bob['height'].values) / 100
        err_string = err_string / 100
        data_bob['error'] = data_bob['error'] / 100
        total_err = np.sqrt(err_string**2 + data_bob['error'].values**2)
        

        total_weighted = DescrStatsW(total_len, weights=1/(total_err)**2)

        total_mean = total_weighted.mean
        total_sigma = total_weighted.std


        self.L_mu = total_mean
        self.L_sigma = total_sigma

        # The error on the mean assuming that each observer is a strong INDEPENDENT woman
        self.L_mu_err = self.L_sigma / np.sqrt(len(self.namelist))

        if self.printing:
            print(f'The length is {self.L_mu} +- {self.L_sigma}')
    
    def get_g(self):

        # Just calculating g, no need for further comments...

        self.g = self.L_mu * (2*np.pi / self.T_mu)**2
        first = 1/(self.T_mu**4) * self.L_sigma**2
        second = 4*self.L_mu**2 / (self.T_mu**6) * self.T_sigma**2
        
        self.g_sigma = 4*np.pi**2 * np.sqrt(np.abs(first - second))

        print(f'The value of g is {self.g} +- {self.g_sigma} m/s')
        
    

# Using the class defined above

# Creating a Pendulum object with a name-list and no single printing
pendulum = Pendulum(['Simone','Niall','Charl'], printing=False)

# Executing the functions on the object to gain g
pendulum.period()
pendulum.length()
pendulum.get_g()



