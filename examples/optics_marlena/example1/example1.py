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
computer = 'ana' # if on mac, else 'anast' if on lenovo

if computer == 'anast':
    DIR = '/Users/anast/OneDrive/Namizje/tns-repo/tns-ana/'
elif computer == 'ana':
    DIR = '/Users/ana/Desktop/tns-ana/'

print('Running on computer: ' + DIR)
sys.path.insert(0, DIR + 'main-programs/')
import tns_module as module

hopping_file = DIR + 'main-programs/parameters-kinetic.txt'
interaction_file = DIR + 'main-programs/parameters-interaction.txt'
perturbation_file = DIR + 'main-programs/parameters-perturbation.txt'
inputs_DIR = DIR + 'examples/optics_marlena/example1/'

''' Add path to input files:
    - input.json: general parameters
    - input_temperature.json: temperatures at which to evaluate transport, spectral width Gamma
    '''

input_file = inputs_DIR + 'input.json'
input_temperature = inputs_DIR + 'input_temperature.json'

''' Add path to output file
results will be saved already during the run, in case the run is not completed '''

Gamma = 0.008
input_optical = {
    "Gamma" : Gamma,
    "deg" : 1000,
    "n_workers" : 1,
    "omega0_low" : 0.01,
    "omega0_high" : 1.0,
    "omega0_len" : 100,
    "space" : "lin",
    "eps" : 1e-5
}


def plot_optics(omega0, results_x, results_y, results_xy, results_yx):
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(8,8))

    # direction x (axis a)
    sigma_x = -results_x['chi_jj0'].imag / omega0
    dsigma_x = -results_x['dchi_jj'].imag / omega0
    ax[0,0].plot(omega0, sigma_x, label='without vertex corrections')
    ax[0,0].plot(omega0, sigma_x + dsigma_x, label='with vertex corrections')
    ax[0,0].set_ylabel(r'$\sigma_x$')

    # direction y (axis c)
    sigma_y = -results_y['chi_jj0'].imag / omega0
    dsigma_y = -results_y['dchi_jj'].imag / omega0
    ax[0,1].plot(omega0, sigma_y, label='without vertex corrections')
    ax[0,1].plot(omega0, sigma_y + dsigma_y, label='with vertex corrections')
    ax[0,1].set_ylabel(r'$\sigma_y$') 

    # mixed xy
    sigma_xy = -results_xy['chi_jj0'].imag / omega0
    dsigma_xy = -results_xy['dchi_jj'].imag / omega0
    ax[1,0].plot(omega0, sigma_xy, label='without vertex corrections')
    ax[1,0].plot(omega0, sigma_xy + dsigma_xy, label='with vertex corrections')
    ax[1,0].set_ylabel(r'$\sigma_{xy}$') 

    # mixed yx
    sigma_yx = -results_yx['chi_jj0'].imag / omega0
    dsigma_yx = -results_yx['dchi_jj'].imag / omega0
    ax[1,1].plot(omega0, sigma_yx, label='without vertex corrections')
    ax[1,1].plot(omega0, sigma_yx + dsigma_yx, label='with vertex corrections')
    ax[1,1].set_ylabel(r'$\sigma_{yx}$')

    for j in range(2):
        for i in range(2):
            ax[j,i].set_xlabel(r'$\omega\,(\text{eV})$')
            ax[j,i].legend()

    plt.show()

####################################################################################################################
## 1. V=0.785 (excitonic), faktor=1, pos=actual positions
s = module.TNS(input_file, hopping_file, interaction_file, perturbation_file,
               rho=None, energije=None, fs=None, vecs=None, fock=None, hartree=None)
pos = 1
s.run_Tdependence(input_temperature)
s.velocities()
omega0, results_x, results_y, results_xy, results_yx = s.optical_responses(input_optical, json_file=False)
colors = np.einsum('ijkl->jkl', np.abs(s.vecs[:4,:,:,:])**2)
Optics = {'omega0' : omega0, 'results_x' : results_x, 'results_y' : results_y, 'results_xy' : results_xy, 'results_yx' : results_yx,
          'energije' : s.energije, 'Nx' : s.Nx, 'Ny' : s.Ny, 'colors' : colors}
plot_optics(omega0, results_x, results_y, results_yx, results_xy)
np.savez(inputs_DIR + f'results/optics_Gamma{Gamma}_Nx{s.Nx}_faktor{s.faktor}_V{s.V}_pos{pos}.npz', **Optics)
####################################################################################################################
## 2. V=0.9 (excitonic), faktor=1, pos=actual positions
s = module.TNS(input_file, hopping_file, interaction_file, perturbation_file,
               rho=None, energije=None, fs=None, vecs=None, fock=None, hartree=None, V=0.9)
pos = 1
s.run_Tdependence(input_temperature)
s.velocities()
omega0, results_x, results_y, results_xy, results_yx = s.optical_responses(input_optical, json_file=False)
colors = np.einsum('ijkl->jkl', np.abs(s.vecs[:4,:,:,:])**2)
Optics = {'omega0' : omega0, 'results_x' : results_x, 'results_y' : results_y, 'results_xy' : results_xy, 'results_yx' : results_yx,
          'energije' : s.energije, 'Nx' : s.Nx, 'Ny' : s.Ny, 'colors' : colors}
plot_optics(omega0, results_x, results_y, results_yx, results_xy)
np.savez(inputs_DIR + f'results/optics_Gamma{Gamma}_Nx{s.Nx}_faktor{s.faktor}_V{s.V}_pos{pos}.npz', **Optics)
####################################################################################################################
## 3. V=0.785 (excitonic), faktor=1, pos=actual positions
pos = 0
s = module.TNS(input_file, hopping_file, interaction_file, perturbation_file,
               rho=None, energije=None, fs=None, vecs=None, fock=None, hartree=None, pos=pos)
s.run_Tdependence(input_temperature)
s.velocities()
omega0, results_x, results_y, results_xy, results_yx = s.optical_responses(input_optical, json_file=False)
colors = np.einsum('ijkl->jkl', np.abs(s.vecs[:4,:,:,:])**2)
Optics = {'omega0' : omega0, 'results_x' : results_x, 'results_y' : results_y, 'results_xy' : results_xy, 'results_yx' : results_yx,
          'energije' : s.energije, 'Nx' : s.Nx, 'Ny' : s.Ny, 'colors' : colors}
plot_optics(omega0, results_x, results_y, results_yx, results_xy)
np.savez(inputs_DIR + f'results/optics_Gamma{Gamma}_Nx{s.Nx}_faktor{s.faktor}_V{s.V}_pos{pos}.npz', **Optics)

####################################################################################################################