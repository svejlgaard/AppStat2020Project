import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, contextlib, sys
from datetime import datetime
from Load import get_angles, get_diameters

# Sets the directory to the current directory
os.chdir(sys.path[0])

time_signature = datetime.now().strftime("%m%d-%H%M")

random_state = 27

plt.style.use('seaborn')

class Ball():
    def __init__(self, name):
        self.name = name
        self.diameter = get_diameters()[self.name]
    
    def get_diameter(self):
        self.diameter_mean = self.diameter.mean()
        self.diameter_mean_err = self.diameter.std()/np.sqrt(len(self.diameter))
        self.diameter_std = self.diameter.std()

    def get_data(self):
        print('Not implemented yet')


# Usage example

small_ball = Ball('SB1')

small_ball.get_diameter()

