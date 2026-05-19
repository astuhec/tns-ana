import os, sys
os.environ['PATH'] = '/Library/TeX/texbin:' + os.environ['PATH']
import numpy as np
from scipy.special import roots_legendre
import matplotlib.pyplot as plt

####################################################################################################################
## This example shows how to run the temperature dependence of DC transport coefficients, Kubo and Boltzmann,
## with or without vertex corrections.
## First, module.TNS is used to initialize the TNS model, based on parameters in input_file.
## Then, method run_Tdependence is used to find rho(T) and mu(T) and to evaluate transport coefficients.
## Temperatures at which to evaluate are based on input_temperature_file, where also the spectral width Gamma is defined.
## Results are saved during the run in file_output, in case the run is not completed (if not all temperatures are evaluated).
## After the run, results are collected and saved in file_output.
####################################################################################################################

''' Add path to programs and parameters '''
computer = 'ana' # if on mac, else 'anast' if on lenovo

if computer == 'anast':
    DIR = '/Users/anast/OneDrive/Namizje/tns-repo/tns-ana/'
elif computer == 'ana':
    DIR = '/Users/ana/Desktop/tns-ana/'

print('Running on computer: ' + DIR)
sys.path.insert(0, DIR + 'main-programs/')
import tns_module as module
import tns_helpers as helpers

hopping_file = DIR + 'main-programs/parameters-kinetic.txt'
interaction_file = DIR + 'main-programs/parameters-interaction.txt'
perturbation_file = DIR + 'main-programs/parameters-perturbation.txt'

''' Add path to input files:
    - input.json: general parameters
    - input_temperature.json: temperatures at which to evaluate transport, spectral width Gamma
    '''
input_file = DIR + 'examples/example_temperature/input.json'
temperature_file = DIR + 'examples/example_temperature/input_temperature.json'

''' Add path to output file '''
file_output = DIR + 'examples/example_temperature/transport_DC_results.npz'

####################################################################################################################

s = module.TNS(input_file, hopping_file, interaction_file, perturbation_file,
               rho=None, energije=None, fs=None, vecs=None, fock=None, hartree=None)

s.run_Tdependence(temperature_file, 
                  save_during=True, file_name=file_output)

results = s.collect_results()
np.savez(file_output, results=results)

####################################################################################################################