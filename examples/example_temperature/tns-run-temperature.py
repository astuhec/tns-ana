import os, sys
os.environ['PATH'] = '/Library/TeX/texbin:' + os.environ['PATH']
import numpy as np
from scipy.special import roots_legendre
import matplotlib.pyplot as plt

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
import tns_tokovi as tokovi

hopping_file = DIR + 'main-programs/parameters-kinetic.txt'
interaction_file = DIR + 'main-programs/parameters-interaction.txt'
perturbation_file = DIR + 'main-programs/parameters-perturbation.txt'

''' Add path to input files:
    - input.json: general parameters
    - input_temperature.json: temperatures at which to evaluate transport, spectral width Gamma
    '''
input_file = DIR + 'examples/example_temperature/input.json'
temperature_file = DIR + 'examples/example_temperature/input_temperature.json'

''' Add path to output file
results will be saved already during the run, in case the run is not completed '''
file_output = DIR + 'examples/example_temperature/transport_DC_results.npz'

####################################################################################################################

s = module.TNS(input_file, hopping_file, interaction_file, perturbation_file,
               rho=None, energije=None, fs=None, vecs=None, fock=None, hartree=None)

''' Add path to tempeature file and run the temperature dependence of transport DC conductivity'''
s.run_Tdependence(temperature_file, 
                  save_during=True, file_name=file_output)
results = s.collect_results()
np.savez(file_output, results=results)

####################################################################################################################