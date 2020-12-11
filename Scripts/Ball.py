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
from statsmodels.stats.weightstats import DescrStatsW
from scipy.stats import chi2_contingency, chi2
from itertools import combinations

# Sets the directory to the current directory
os.chdir(sys.path[0])

time_signature = datetime.now().strftime("%m%d-%H%M")

random_state = 27

plt.style.use('seaborn')

class Ball():
    def __init__(self, name):
        # The name should be either 'SB1' (small ball) or 'LB1' (large ball)
        self.name = name

        # Importing the diameter of the given ball
        self.diameter = get_diameters()[self.name]

        # Converting to the naming sheme of the other group
        if self.name == 'SB1':
            self.ball = 'mid'
        if self.name == 'LB1':
            self.ball = 'big'
    
    def get_chi2(self, name, measurements, err):
        chi2_sum = [(i - j)**2 / err[np.argwhere(measurements == j)]**2 for i,j in combinations(measurements,2)]
        true_chi2 = np.sum(chi2_sum)
        true_p = chi2.sf(true_chi2, len(measurements)-1)
        print(f'For the {name} measurements, the Chi2 is {true_chi2:.3f} and p is {true_p:.3f}')
    
    def get_peaks(self, orientation):

        self.peak_dict = dict()
        # Orientation should be 'L' (left) or 'R' (right)
        self.orientation = orientation

        for o in ['L','R']:
            import glob
            files = glob.glob(f'../Data/BallIncline/*{self.ball}_{o}.csv')
            
            time_list = list()
            for f in files:
                # Load the data from the relevant file
                dat = np.genfromtxt(f, delimiter=',', skip_header=13, names=True)
                time = dat['Time_s']
                voltage = dat['Channel_1_V']
                # Using the scipy find_peaks with a minimum height of 4 to avoid detecting noise as peaks
                # and a minimum distance of 100 to avoid detecting the same peaks twice
                peaks, _ = find_peaks(voltage, height=4, distance=100)
                assert len(peaks) == 5
                t = time[peaks]
                time_list.append(t)
            time_list = np.array(time_list)

            # To 'calibrate' the detections to go though 0 and avoid different cutoffs 
            time_list = time_list - time_list[:,0,None]

            # Saving and calculating error on mean
            times = np.mean(time_list,axis=0)
            times_err = np.std(time_list, axis=0) / np.sqrt(3)

            self.peak_dict.update({f'peak_{o}': [times, times_err]})
        
        self.times = self.peak_dict[f'peak_{self.orientation}'][0]
        self.times_err = self.peak_dict[f'peak_{self.orientation}'][1]
            
        return self.times

    def get_diameter(self):
        self.get_chi2(f'Diameter', self.diameter.values, np.array([0.05, 0.05, 0.05]))
        self.diameter_mean = self.diameter.mean()
        self.diameter_mean_err = self.diameter.std()/np.sqrt(len(self.diameter))
        self.diameter_std = self.diameter.std()
        print(f' The diameter of the {self.ball} is {1000*self.diameter_mean:.2f} +- {1000*self.diameter_mean_err:.2f}')


    def get_acc(self):
        lengths = pd.read_csv('../Data/BallIncline/rail_lengths.txt',sep=',', header=3)


        # To get the measures from the folding ruler, I use the index :5
        fold_measure = lengths.iloc[:5,:]

        # The pandas dataframe read_csv adds extra columns without any numbers in them. 
        # This removes those columns
        fold_measure = fold_measure.reset_index(drop=True)

        # Similar for the ruler, but it has index from line 5 and up (5:)
        ruler_measure = lengths.iloc[5:,:]
        ruler_measure = ruler_measure.reset_index(drop=True)
        
        # Combining them under the assumption that both observers and equipment are independent
        length_measure = pd.concat([fold_measure, ruler_measure], axis=1, copy=True)

        # Converting from cm to m
        length_measure = length_measure / 100
        self.length_mu = length_measure.mean(axis=1).to_numpy()
        self.length_std = length_measure.std(axis=1).to_numpy()

        # The error on the mean, where the np.sqrt(6) is from the independence assumption
        self.length_mu_err = self.length_std/ np.sqrt(6)
        
        # Assuming uniform acceleration
        def s(t, a, b, c):
            return 1/2*a*t*t + b*t + c


        self.a_dict = dict()

        for o in ['L', 'R']:

            # Fitting via chi2
            acc_fit = Chi2Regression(f=s, x=self.peak_dict[f'peak_{o}'][0], y=self.length_mu, error=self.length_mu_err)
            position_fit = Minuit(acc_fit, pedantic=False, a=2.0, b=0.0, c=self.length_mu[0])
            position_fit.migrad()

            # Saving the acceleration value
            self.a_dict.update({f'a_{o}': [position_fit.values["a"], position_fit.errors["a"]]})
            
            if o == self.orientation:
                self.a = position_fit.values["a"]
                self.a_err = position_fit.errors["a"]

            # Saving the result as a figure
            plt.figure(f'Position_{self.ball}_{o}')
            plt.errorbar(self.peak_dict[f'peak_{o}'][0], self.length_mu, yerr=self.length_mu_err, xerr=self.peak_dict[f'peak_{o}'][1], fmt='.', label='Data')
            plt.plot(self.peak_dict[f'peak_{o}'][0], s(self.peak_dict[f'peak_{o}'][0] , *position_fit.args), label=f'Fit with a = {position_fit.values["a"]:.2f} +- {position_fit.errors["a"]:.2f}')
            plt.xlabel('Time (s)')
            plt.ylabel('Position (m)')
            plt.legend()
            plt.savefig(f'../Figures/BallIncline/Acceleration_plot_{self.ball}_{o}.pdf')


        # Saving the result as a pickle file as requested by Emma
        data_final = open(f'../Data/BallIncline/Acceleration_plot_{self.ball}_{self.orientation}_results.pkl','wb')

        dict_final = {'x': self.times,
                    'y': self.length_mu,
                    'y_err': self.length_mu_err,
                    }
        
        for param in position_fit.values:
            dict_final.update({f'fit_{param}': position_fit.values[param]})
            dict_final.update({f'fit_{param}_err': position_fit.errors[param]})

        import pickle
        pickle.dump(dict_final, data_final, protocol=2)

        data_final.close()
    

    def get_angle(self, flip = True):
        angles = get_angles()

        angle_dict = dict()
        for a in ['r','l']:
            phi = angles[[f'inc_{a}']]
            flip = angles[[f'inc_flip_{a}']]
            err = 1

            phi_des = DescrStatsW(phi, weights = [err, err, err])
            phi_mean = phi_des.mean
            phi_mean_err = phi_des.std / np.sqrt(3)
            
            flip_des = DescrStatsW(flip, weights = [err, err, err])
            flip_mean = flip_des.mean
            flip_mean_err = flip_des.std / np.sqrt(3)

            true_phi = (phi_mean + flip_mean) / 2

            true_phi_err = np.sqrt(phi_mean_err**2 + flip_mean_err**2) / 2
            
            angle_dict.update({f'phi_{a}': [true_phi, true_phi_err]})
        

        self.phi_r_mu = angle_dict['phi_r'][0]
        self.phi_r_std = angle_dict['phi_r'][1]


        self.phi_l_mu = angle_dict['phi_l'][0]
        self.phi_l_std = angle_dict['phi_l'][1]

        self.dphi_mu_g = (self.phi_r_mu - self.phi_l_mu) / 2
        self.dphi_std_g = np.sqrt(self.phi_r_std**2 + self.phi_l_std**2)
        self.dphi_mu_err_g = self.dphi_std_g / np.sqrt(2) + 0.05


        upper = (self.a_dict['a_L'][0] - self.a_dict['a_R'][0]) * np.sin(np.radians(angle_dict[f'phi_{self.orientation.lower()}'][0]))
        lower = (self.a_dict['a_L'][0] + self.a_dict['a_R'][0]) * np.cos(np.radians(angle_dict[f'phi_{self.orientation.lower()}'][0]))
        
        self.dphi_mu_a = upper / lower
        
        x = np.radians(angle_dict[f'phi_{self.orientation.lower()}'][0])
        L = self.a_dict['a_L'][0]
        R = self.a_dict['a_R'][0]

        err1 = (np.sin(x)/((L+R)*np.cos(x))-(L-R)*np.sin(x)/((L+R)**2*np.cos(x)))**2 * self.phi_l_std**2
        err2 = (-np.sin(x)/((L+R)*np.cos(x))-(L-R)*np.sin(x)/((L+R)**2*np.cos(x)))**2 * self.phi_r_std**2
        err3 = ((L-R)/(L+R)+(L-R)*np.sin(x)**2/((L+R)*np.cos(x)**2))**2 * angle_dict[f'phi_{self.orientation.lower()}'][1]**2
        
        self.dphi_std_a = np.sqrt(err1 + err2 + err3) + 0.05
        
        combined_dphi = DescrStatsW([self.dphi_mu_g, self.dphi_mu_a], [self.dphi_mu_err_g, self.dphi_std_a])


        self.dphi_mu = combined_dphi.mean
        self.dphi_std = combined_dphi.std / np.sqrt(2)

        self.get_chi2(f'dphi_{self.orientation}', np.array([self.dphi_mu_a,self.dphi_mu_g]), np.array([self.dphi_std_a, self.dphi_mu_err_g]))
        

    def get_g(self):
        rail = np.loadtxt('../Data/BallIncline/Rail_diameter.txt')

        # Converting from mm to m
        rail = rail / 1000

        self.rail_mu = np.mean(rail)
        self.rail_std = np.std(rail)
        self.rail_mu_err = self.rail_std / np.sqrt(3)

        # Based on the equations from the report for the ball 
        paranthesis = 1 + ( (2/5) * (self.diameter_mean**2) / (self.diameter_mean**2 - self.rail_mu**2) )
        if self.orientation == 'L':
            angle = np.radians(self.phi_l_mu + self.dphi_mu)
        if self.orientation == 'R':
            angle = np.radians(self.phi_r_mu - self.dphi_mu)
        self.g = paranthesis * self.a / np.sin(angle)
        self.g = self.g[0]
        
        # Splitted to 
        err_term1 = paranthesis * 1/np.sin(angle) * self.a_err**2
        err_term2 = self.a**2 * paranthesis * np.cos(angle)**2 / np.sin(angle)**4 * np.radians(self.phi_r_std)**2
        err_term3 = self.a**2 * paranthesis * np.cos(angle)**2 / np.sin(angle)**4 * np.radians(self.dphi_std)**2
        err_term4 = self.a**2 / np.sin(angle)**2 * (4*self.diameter_mean / (5* (self.diameter_mean**2 + self.rail_mu**2)) - 4*self.diameter_mean**3 / (5* (self.diameter_mean**2 + self.rail_mu**2)))**2 * self.diameter_std**2
        err_term5 = self.a**2 * 16 * self.diameter_mean**4 * self.rail_mu**2 / (25 * np.sin(angle)**2 * (self.diameter_mean**2 - self.rail_mu**2)**4 ) * self.rail_std**2 

        self.g_err = np.sqrt(err_term1 + err_term2 + err_term3 + err_term4 + err_term5)
        self.g_err = self.g_err[0]




        



# Usage example


for size in ['SB1','LB1']:
    small_ball = Ball(size)

    small_ball.get_diameter()

    for o in ['R', 'L']:

        small_ball.get_peaks(o)

        small_ball.get_acc()

        small_ball.get_angle()

        small_ball.get_g()

        print(f'We find g = {small_ball.g:.2f} +- {small_ball.g_err:.2f} m/s for {small_ball.ball} ball oriented {small_ball.orientation}')