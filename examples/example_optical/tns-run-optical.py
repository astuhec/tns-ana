import os, sys
os.environ['PATH'] = '/Library/TeX/texbin:' + os.environ['PATH']
import numpy as np
from scipy.special import roots_legendre
import matplotlib.pyplot as plt

####################################################################################################################

''' Add path to programs '''
sys.path.insert(0, '/Users/anast/OneDrive/Namizje/tns-repo/tns-ana/main-programs/')
import tns_module as module
import tns_helpers as helpers
import tns_tokovi as tokovi

''' Add path to parameters '''
hopping_file = "/Users/anast/OneDrive/Namizje/tns-repo/tns-ana/main-programs/parameters-kinetic.txt"
interaction_file = "/Users/anast/OneDrive/Namizje/tns-repo/tns-ana/main-programs/parameters-interaction.txt"
perturbation_file = "/Users/anast/OneDrive/Namizje/tns-repo/tns-ana/main-programs/parameters-perturbation.txt"

''' Add path to input files:
    - input.json: general parameters
    - input_temperature.json: temperatures at which to evaluate transport, spectral width Gamma
    '''
input_file = "/Users/anast/OneDrive/Namizje/tns-repo/tns-ana/examples/input.json"
temperature_file = "/Users/anast/OneDrive/Namizje/tns-repo/tns-ana/examples/input_temperature.json"

''' Add path to output file
results will be saved already during the run, in case the run is not completed '''
file_output = "/Users/anast/OneDrive/Namizje/tns-repo/tns-ana/examples/transport_DC_results.npz"

####################################################################################################################

s = module.TNS(input_file, hopping_file, interaction_file, perturbation_file,
               rho=None, energije=None, fs=None, vecs=None, fock=None, hartree=None)

''' Add path to tempeature file and run the temperature dependence of transport DC conductivity'''
s.run_Tdependence(temperature_file, 
                  save_during=True, file_name=file_output)
results = s.collect_results()
np.savez(file_output, results=results)

####################################################################################################################