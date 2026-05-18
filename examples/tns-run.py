import os, sys
os.environ['PATH'] = '/Library/TeX/texbin:' + os.environ['PATH']
import numpy as np
from scipy.special import roots_legendre
import matplotlib.pyplot as plt

sys.path.insert(0, '/Users/anast/OneDrive/Namizje/tns-repo/tns-ana/main-programs/')

import tns_module as module
import tns_helpers as helpers
import tns_tokovi as tokovi

#input_file = "/Users/ana/Desktop/tns-ana/examples/input.txt"
#hopping_file = "/Users/ana/Desktop/tns-ana/main-programs/parametri-kinetic.txt"
#interaction_file = "/Users/ana/Desktop/tns-ana/main-programs/parameters-interaction.txt"

hopping_file = "/Users/anast/OneDrive/Namizje/tns-repo/tns-ana/main-programs/parameters-kinetic.txt"
interaction_file = "/Users/anast/OneDrive/Namizje/tns-repo/tns-ana/main-programs/parameters-interaction.txt"
perturbation_file = "/Users/anast/OneDrive/Namizje/tns-repo/tns-ana/main-programs/parameters-perturbation.txt"

input_file = "/Users/anast/OneDrive/Namizje/tns-repo/tns-ana/examples/input.json"
temperature_file = "/Users/anast/OneDrive/Namizje/tns-repo/tns-ana/examples/input_temperature.json"

file_output = "/Users/anast/OneDrive/Namizje/tns-repo/tns-ana/examples/transport_DC_results.npz"

s = module.TNS(input_file, hopping_file, interaction_file, perturbation_file,
               rho=None, energije=None, fs=None, vecs=None, fock=None, hartree=None)

s.run_Tdependence(temperature_file, evaluate_transport_DC=True, evaluate_vertex_DC=False, 
                  save_during=True, file_name=file_output)
 
omega0 = np.linspace(0.01,0.8,50)
Gamma = 0.008
mu_ = s.mu / Gamma
invt = Gamma / s.Ts[-1]
deg = 200
nodes, weights = roots_legendre(deg)
n_workers = 1

rho_tilde_cache = {}
rho_tilde_factory = tokovi.make_rho_tilde_factory(s.interaction, s.a, s.b, s.kymesh, s.kxmesh, s.vecs)
eps = 1e-4
results_x, results_y = tokovi.compute_chi(omega0, Gamma, mu_, invt, nodes, weights, s.thetas, s.parities, s.tok_x1, s.mat_x, s.tok_y1, s.mat_y, s.energije, rho_tilde_factory, eps=eps, n_workers=n_workers, verbose=True)

plt.plot(omega0, -results_y['chi_jj0'].imag / omega0)
plt.plot(omega0, -results_y['chi_jj0'].imag / omega0 - results_y['dchi_jj'].imag / omega0)
plt.show()