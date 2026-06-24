import os, sys, json
os.environ['PATH'] = '/Library/TeX/texbin:' + os.environ['PATH']
import numpy as np

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
import tns_helpers as helpers
import plotting

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
input_optical_0785 = {
    "Gamma" : Gamma,
    "deg" : 1000,
    "n_workers" : 1,
    "omega0_low" : 0.01,
    "omega0_high" : 0.2,
    "omega0_len" : 100,
    "space" : "lin",
    "eps" : 1e-5
}

input_optical_09 = {
    "Gamma" : Gamma,
    "deg" : 1000,
    "n_workers" : 1,
    "omega0_low" : 0.01,
    "omega0_high" : 0.8,
    "omega0_len" : 100,
    "space" : "lin",
    "eps" : 1e-5
}

####################################################################################################################
## 1. V=0.785 (excitonic), faktor=1, pos=actual positions
## phi neq 0, Deltas neq 0
s = module.TNS(input_file, hopping_file, interaction_file, perturbation_file,
               rho=None, energije=None, fs=None, vecs=None, fock=None, hartree=None)
pos = 1
s.run_Tdependence(input_temperature)
colors = np.einsum('ijkl->jkl', np.abs(s.vecs[:4,:,:,:])**2)
#plotting.bands(s.energije, colors, s.mu, name=f'results/bands_Nx{s.Nx}_faktor{s.faktor}_V{s.V}_U{s.U}')

s.velocities()
omega0, results_x, results_y, results_xy, results_yx = s.optical_responses(input_optical_0785, json_file=False)
Optics = {'omega0' : omega0, 'results_x' : results_x, 'results_y' : results_y, 'results_xy' : results_xy, 'results_yx' : results_yx,
          'energije' : s.energije, 'Nx' : s.Nx, 'Ny' : s.Ny, 'colors' : colors}
#plotting.optics(omega0, results_x, results_y, results_yx, results_xy, name=f'results/optics_Gamma{Gamma}_Nx{s.Nx}_faktor{s.faktor}_V{s.V}_pos{pos}')
np.savez(inputs_DIR + f'results/optics_small_Gamma{Gamma}_Nx{s.Nx}_faktor{s.faktor}_V{s.V}_pos{pos}.npz', **Optics)

####################################################################################################################
## 2. V=0.9 (excitonic), faktor=1, pos=actual positions
## phi = 0 but Deltas are not zero, i.e. Fock is not zero
s = module.TNS(input_file, hopping_file, interaction_file, perturbation_file,
               rho=None, energije=None, fs=None, vecs=None, fock=None, hartree=None, V=0.9, U=2.5)
pos = 1
s.run_Tdependence(input_temperature)
colors = np.einsum('ijkl->jkl', np.abs(s.vecs[:4,:,:,:])**2)

plotting.bands(s.energije, colors, s.mu, name=f'results/bands_Nx{s.Nx}_faktor{s.faktor}_V{s.V}_U{s.U}')

s.velocities()
omega0, results_x, results_y, results_xy, results_yx = s.optical_responses(input_optical_09, json_file=False)
Optics = {'omega0' : omega0, 'results_x' : results_x, 'results_y' : results_y, 'results_xy' : results_xy, 'results_yx' : results_yx,
          'energije' : s.energije, 'Nx' : s.Nx, 'Ny' : s.Ny, 'colors' : colors}
plotting.optics(omega0, results_x, results_y, results_yx, results_xy, name=f'results/optics_Gamma{Gamma}_Nx{s.Nx}_faktor{s.faktor}_V{s.V}_pos{pos}')
np.savez(inputs_DIR + f'results/optics_small_Gamma{Gamma}_Nx{s.Nx}_faktor{s.faktor}_V{s.V}_pos{pos}.npz', **Optics)

####################################################################################################################
## 3. V=0.785 (excitonic), faktor=1, pos=0
## phi neq 0, Deltas neq 0
pos = 0
s = module.TNS(input_file, hopping_file, interaction_file, perturbation_file,
               rho=None, energije=None, fs=None, vecs=None, fock=None, hartree=None, pos=pos)
s.run_Tdependence(input_temperature)
colors = np.einsum('ijkl->jkl', np.abs(s.vecs[:4,:,:,:])**2)
plotting.bands(s.energije, colors, s.mu, name=f'results/bands_Nx{s.Nx}_faktor{s.faktor}_V{s.V}_U{s.U}')

s.velocities()
omega0, results_x, results_y, results_xy, results_yx = s.optical_responses(input_optical_0785, json_file=False)
Optics = {'omega0' : omega0, 'results_x' : results_x, 'results_y' : results_y, 'results_xy' : results_xy, 'results_yx' : results_yx,
          'energije' : s.energije, 'Nx' : s.Nx, 'Ny' : s.Ny, 'colors' : colors}
plotting.optics(omega0, results_x, results_y, results_yx, results_xy, name=f'results/optics_Gamma{Gamma}_Nx{s.Nx}_faktor{s.faktor}_V{s.V}_pos{pos}')
np.savez(inputs_DIR + f'results/optics_small_Gamma{Gamma}_Nx{s.Nx}_faktor{s.faktor}_V{s.V}_pos{pos}.npz', **Optics)

####################################################################################################################
## 4. V=0.785 (excitonic), faktor=0, pos actual positions
## phi = 0, Deltas = 0
pos = 1
s = module.TNS(input_file, hopping_file, interaction_file, perturbation_file,
               rho=None, energije=None, fs=None, vecs=None, fock=None, hartree=None, faktor=0.0)
s.run_Tdependence(input_temperature)
colors = np.einsum('ijkl->jkl', np.abs(s.vecs[:4,:,:,:])**2)
plotting.bands(s.energije, colors, s.mu, name=f'results/bands_Nx{s.Nx}_faktor{s.faktor}_V{s.V}_U{s.U}')

s.velocities()
omega0, results_x, results_y, results_xy, results_yx = s.optical_responses(input_optical_0785, json_file=False)
Optics = {'omega0' : omega0, 'results_x' : results_x, 'results_y' : results_y, 'results_xy' : results_xy, 'results_yx' : results_yx,
          'energije' : s.energije, 'Nx' : s.Nx, 'Ny' : s.Ny, 'colors' : colors}
plotting.optics(omega0, results_x, results_y, results_yx, results_xy, name=f'results/optics_Gamma{Gamma}_Nx{s.Nx}_faktor{s.faktor}_V{s.V}_pos{pos}')
np.savez(inputs_DIR + f'results/optics_small_Gamma{Gamma}_Nx{s.Nx}_faktor{s.faktor}_V{s.V}_pos{pos}.npz', **Optics)

####################################################################################################################
## 5. V=0.9 (non-excitonic), faktor=0, pos actual positions
pos = 1
s = module.TNS(input_file, hopping_file, interaction_file, perturbation_file,
               rho=None, energije=None, fs=None, vecs=None, fock=None, hartree=None, faktor=0.0, V=0.9)
s.run_Tdependence(input_temperature)
colors = np.einsum('ijkl->jkl', np.abs(s.vecs[:4,:,:,:])**2)
plotting.bands(s.energije, colors, s.mu, name=f'results/bands_Nx{s.Nx}_faktor{s.faktor}_V{s.V}_U{s.U}')

s.velocities()
omega0, results_x, results_y, results_xy, results_yx = s.optical_responses(input_optical_09, json_file=False)
Optics = {'omega0' : omega0, 'results_x' : results_x, 'results_y' : results_y, 'results_xy' : results_xy, 'results_yx' : results_yx,
          'energije' : s.energije, 'Nx' : s.Nx, 'Ny' : s.Ny, 'colors' : colors}
plotting.optics(omega0, results_x, results_y, results_yx, results_xy, name=f'results/optics_Gamma{Gamma}_Nx{s.Nx}_faktor{s.faktor}_V{s.V}_pos{pos}')
np.savez(inputs_DIR + f'results/optics_small_Gamma{Gamma}_Nx{s.Nx}_faktor{s.faktor}_V{s.V}_pos{pos}.npz', **Optics)