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
computer = 'anast' # if on mac, else 'anast' if on lenovo

if computer == 'anast':
    DIR = '/Users/anast/OneDrive/Namizje/tns-ana/'
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

''' Add path to output file '''
file_output = DIR + 'examples/example_temperature/transport_DC_results.npz'

####################################################################################################################

s = module.TNS(input_file, hopping_file, interaction_file, perturbation_file,
               rho=None, energije=None, fs=None, vecs=None, fock=None, hartree=None)

np.save('vecs.npy', s.vecs)
np.save('kymesh.npy', s.kymesh)
np.save('kxmesh.npy', s.kxmesh)
np.save('kxmesh.npy', s.energije)

print(s.mu)

'''thetas = s.thetas
Nop = len(thetas)
vecs = s.vecs
energije = s.energije
Ny = s.Ny
Nx = s.Nx
mu = s.mu

rhos = tokovi.rho_operators(Nop, s.kymesh, s.kxmesh, s.interaction, s.a, s.b)

T = 1/100

rho_expectationvalues = np.zeros(Nop, dtype=np.complex128)
fermi_dirac = 1 / (np.exp((energije - mu) / T) + 1)

for i in range(Nop):
    rho_tilde = tokovi.operator_tilde(rhos[i], vecs)
    for m in range(Ny):
        for n in range(s.Nx):
            rho_expectationvalues[i] += np.sum(np.diag(rho_tilde[:,:,m,n]) * fermi_dirac[:,m,n])

    print(i, rho_expectationvalues[i])

np.save('./results/rho_expects.npy', rho_expectationvalues)'''
'''np.save('../../compare-denis/example1/hk_ana.npy', s.hop)
np.save('../../compare-denis/example1/rho_ana.npy', s.rho)
np.save('../../compare-denis/example1/hartree_ana.npy', s.hartree)
np.save('../../compare-denis/example1/fock_ana.npy', s.fock)
np.save('../../compare-denis/example1/hartree_list.npy', s.hartree_list)
np.save('../../compare-denis/example1/Kxmesh.npy', s.kxmesh)
'''
#s.run_Tdependence(temperature_file, 
#                  save_during=True, file_name=file_output)

#results = s.collect_results()
#np.savez(file_output, results=results)

####################################################################################################################