import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, contextlib, sys
from datetime import datetime
from Load import get_angles, get_diameters
from scipy.signal import find_peaks
from iminuit import Minuit 
from probfit import Chi2Regression
from scipy.optimize import curve_fit

# Sets the directory to the current directory
os.chdir(sys.path[0])

time_signature = datetime.now().strftime("%m%d-%H%M")

random_state = 27

plt.style.use('seaborn')

class Ball():
    def __init__(self, name):
        self.name = name
        self.diameter = get_diameters()[self.name]
    
    def get_peaks(self,no_jump = True, flipped = True):
        import glob
        filename = f'../Data/BallIncline/{self.name}'
        if no_jump and flipped:
            filename += '_flipped_nojump.csv' 
        if no_jump and not flipped:
            filename += '_nojump.csv'
        if not no_jump and flipped:
            filename += '_flipped.csv'
        if not no_jump and not flipped:
            filename += '.csv'
        filename = '../Data/ball7_8mm_example.csv'
        dat = np.genfromtxt(filename, delimiter=',', skip_header=13, names=True)
        time = dat['Time_s']
        voltage = dat['Channel_1_V']
        peaks, _ = find_peaks(voltage, height=4, distance=100)
        assert len(peaks) == 5
        t = time[peaks]
        self.times = t
        return t

    def get_diameter(self):
        self.diameter_mean = self.diameter.mean()
        self.diameter_mean_err = self.diameter.std()/np.sqrt(len(self.diameter))
        self.diameter_std = self.diameter.std()


    def get_acc(self):
        lengths = pd.read_csv('../Data/BallIncline/rail_lengths.txt',sep=',', header=3)

        fold_measure = lengths.iloc[:5,:]
        fold_measure = fold_measure.reset_index(drop=True)

        ruler_measure = lengths.iloc[5:,:]
        ruler_measure = ruler_measure.reset_index(drop=True)
    
        
        length_measure = pd.concat([fold_measure, ruler_measure], axis=1, copy=True)
        length_measure = length_measure / 100
        self.length_mu = length_measure.mean(axis=1).to_numpy()
        self.length_std = length_measure.std(axis=1).to_numpy()
        self.length_mu_err = self.length_std/ np.sqrt(6)
        
        
        def s(t, a, b, c):
            return 1/2*a*t*t + b*t + c


        # Trying to use iMinuit...
        acc_fit = Chi2Regression(f=s, x=self.times, y=self.length_mu, error=self.length_mu_err)
        position_fit = Minuit(acc_fit, pedantic=False, a=2.0, b=0.0, c=self.length_mu[0])
        position_fit.migrad()

        plt.figure('Position')
        plt.errorbar(self.times, self.length_mu, yerr=self.length_mu_err, fmt='.', label='Data')
        plt.plot(self.times, s(self.times , *position_fit.args), label=f'Fit with a = {position_fit.values["a"]:.2f} +- {position_fit.errors["a"]:.2f}')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.legend()
        plt.savefig('../Figures/BallIncline/Acceleration_plot.pdf')

        self.a = position_fit.values["a"]
        self.a_err = position_fit.errors["a"]
    

    def get_angle(self):
        angles = get_angles()
        phi_r = angles[['inc_r', 'inc_flip_r']]
        phi_l = angles[['inc_l','inc_flip_l']]

        phi_r = phi_r.mean(axis=0)
        phi_l = phi_l.mean(axis=0)

        self.phi_r_mu = phi_r.mean()
        self.phi_r_std = phi_r.std()
        self.phi_r_mu_err = self.phi_r_std / np.sqrt(3)

        self.phi_l_mu = phi_l.mean()
        self.phi_l_std = phi_l.std()
        self.phi_l_mu_err = self.phi_l_std / np.sqrt(3)

        self.dphi_mu = self.phi_r_mu - self.phi_l_mu
        self.dphi_std = self.phi_r_std + self.phi_l_std
        self.dphi_mu_err = self.dphi_std / np.sqrt(2)

    def get_g(self):
        rail = np.loadtxt('../Data/BallIncline/Rail_diameter.txt')
        rail = rail / 1000
        self.rail_mu = np.mean(rail)
        self.rail_std = np.std(rail)
        self.rail_mu_err = self.rail_std / np.sqrt(3)
        paranthesis = 1 + ( (2/5) * (self.diameter_mean**2) / (self.diameter_mean**2 - self.rail_mu**2) )
        
        self.g = paranthesis * self.a / np.sin(self.phi_r_mu - self.dphi_mu)
        
        err_term1 = paranthesis * 1/np.sin(self.phi_r_mu - self.dphi_mu) * self.a_err**2
        err_term2 = self.a**2 * paranthesis * np.cos(self.phi_r_mu - self.dphi_mu)**2 / np.sin(self.phi_r_mu - self.dphi_mu)**4 * self.phi_r_std
        err_term3 = self.a**2 * paranthesis * np.cos(self.phi_r_mu - self.dphi_mu)**2 / np.sin(self.phi_r_mu - self.dphi_mu)**4 * self.dphi_std
        err_term4 = self.a**2 / np.sin(self.phi_r_mu - self.dphi_mu)**2 * (4*self.diameter_mean / (5* (self.diameter_mean**2 + self.rail_mu**2)) - 4*self.diameter_mean**3 / (5* (self.diameter_mean**2 + self.rail_mu**2)))**2 * self.diameter_std**2
        err_term5 = self.a**2 * 16 * self.diameter_mean**4 * self.rail_mu**2 / (25 * np.sin(self.phi_r_mu - self.dphi_mu)**2 * (self.diameter_mean**2 - self.rail_mu**2)**4 ) * self.rail_std**2 

        self.g_err = np.sqrt(err_term1 + err_term2 + err_term3 + err_term4 + err_term5)




        



# Usage example

small_ball = Ball('SB1')

small_ball.get_diameter()

small_ball.get_peaks()

small_ball.get_acc()

small_ball.get_angle()

small_ball.get_g()

print(f'We find g = {small_ball.g:.2f} +- {small_ball.g_err:.2f} m/s')