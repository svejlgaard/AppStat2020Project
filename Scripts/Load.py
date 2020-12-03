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


def get_diameters():
    import glob
    all_files = glob.glob('../Data/BallIncline/*diameter.txt')

    file_list = []

    for filename in all_files:
        data = pd.read_csv(filename)
        data = data.dropna(axis=1)
        name = filename.split("/")[-1]
        data.columns = [f'{name.split("_")[0]}']
        data.index = ['Simone', 'Niall', 'Charl']
        file_list.append(data / 1000)
    
    data = pd.concat(file_list, axis=1)

    return data

def get_angles():
    import glob
    filename = glob.glob('../Data/BallIncline/*Incline.txt')[0]
    data = pd.read_csv(filename, header=1)
    data = data.dropna(axis=1)
    data = data.transpose()
    data.index = ['Simone', 'Niall', 'Charl']
    data.columns = ['inc_r','inc_l','inc_flip_r','inc_flip_l']
    return data
    

#How to access the data

#get_diameters()

#get_angles()


#def load(filename, experiment):
#    exp_list = ['BallIncline','Pendulum']

#   if experiment not in exp_list:
#        print(f'The chosen experiment could not be found. Please choose from [{exp_list}]')
    
#    filetype = filename.split('.')[1]

#    if filetype == 'txt':
#        data = pd.read_csv(f'../Data/{experiment}/{filename}', sep=" ", skiprows=1)
    
#    return data