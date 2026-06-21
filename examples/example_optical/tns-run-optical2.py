import os, sys, json
os.environ['PATH'] = '/Library/TeX/texbin:' + os.environ['PATH']
import numpy as np
from scipy.special import roots_legendre
import matplotlib.pyplot as plt

####################################################################################################################
## This example shows how to obtain optical conductivity, with or without vertex corrections.
## First, module.TNS is used to initialize the TNS model, based on parameters in input_file.
## Then, method run_Tdependence is used to reach some temperature T, based on input_temperature_file.
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

hopping_file = DIR + 'main-programs/parameters-kinetic.txt'
interaction_file = DIR + 'main-programs/parameters-interaction.txt'
perturbation_file = DIR + 'main-programs/parameters-perturbation.txt'

''' Add path to input files:
    - input.json: general parameters
    - input_temperature.json: temperatures at which to evaluate transport, spectral width Gamma
    '''
input_file = DIR + 'examples/example_optical/input.json'
input_temperature = DIR + 'examples/example_optical/input_temperature.json'
input_optical = DIR + 'examples/example_optical/input_optical.json'

''' Add path to output file
results will be saved already during the run, in case the run is not completed '''
file_output = DIR + 'examples/example_optical/optical_results.npz'

####################################################################################################################
## Find equilibrium state at desired temperature T.
s = module.TNS(input_file, hopping_file, interaction_file, perturbation_file,
               rho=None, energije=None, fs=None, vecs=None, fock=None, hartree=None)

s.run_Tdependence(input_temperature)

####################################################################################################################
## Update current vertices, rotated to the band basis.
s.velocities()

## Loop over Gammas
Gamma = 0.008

input_optical = {
    "Gamma" : Gamma,
    "deg" : 1000,
    "n_workers" : 1,
    "omega0_low" : 0.01,
    "omega0_high" : 1.0,
    "omega0_len" : 150,
    "space" : "lin",
    "eps" : 1e-5
}

omega0, results_x, results_y, results_xy, results_yx = s.optical_responses(input_optical, json_file=False)

Optics = {'omega0' : omega0, 'results_x' : results_x, 'results_y' : results_y}
#np.savez(DIR + f'examples/example_optical/results/optics2_Gamma{Gamma}_Nx{s.Nx}.npz', **Optics)

plot_conductivity = True

a = 3.51 # A
b = 15.79 # A
c = 13.42 # A
V0 = a * b * c
G0 = 25.8 * 1e3 # 1/Ohm

prefactor = 4 * np.pi * 1e10 / V0 / G0 # prefactor to get conductivity into right units, i.e. 1/Ohm m

if plot_conductivity:
    fig, ax = plt.subplots(ncols=2, figsize=(8,4))

    # direction x (axis a)
    sigma_x = -results_xy['chi_jj0'].imag / omega0 * prefactor
    dsigma_x = -results_xy['dchi_jj'].imag / omega0 * prefactor
    ax[0].plot(omega0, sigma_x, label='without vertex corrections')
    ax[0].plot(omega0, sigma_x + dsigma_x, label='with vertex corrections')
    ax[0].set_xlabel(r'$\omega\,(\text{eV})$')

    # direction y (axis b)
    sigma_y = -results_yx['chi_jj0'].imag / omega0 * prefactor
    dsigma_y = -results_yx['dchi_jj'].imag / omega0 * prefactor
    ax[1].plot(omega0, sigma_y, label='without vertex corrections')
    ax[1].plot(omega0, sigma_y + dsigma_y, label='with vertex corrections')
    ax[1].set_xlabel(r'$\omega\,(\text{eV})$')

    ax[0].set_ylabel(r'$\sigma_x$')
    ax[1].set_ylabel(r'$\sigma_y$') 
    plt.show()
####################################################################################################################