import os, sys
os.environ['PATH'] = '/Library/TeX/texbin:' + os.environ['PATH']
sys.path.append("/Users/ana/Desktop/tns-ana/main-programs/")
import numpy as np
from scipy.special import roots_legendre
import matplotlib.pyplot as plt

import os
import tns_module as module
import tns_helpers as helpers
import tns_tokovi as tokovi

input_file = "/Users/ana/Desktop/tns-ana/examples/input.txt"

s = module.TNS(input_file)

beta0 = 150
scale = 1.005
Gammas = [0.008]
betas = beta0 / scale**np.arange(1,11)
stops = [len(betas) - 1]
parameters = {'Nomega' : 5000,
              'eps' : 1e-4,
              'deg' : 500,
              'eps2' : 1e-4,
              'omega0' : np.logspace(-7,-2,50),
              'n_workers' : 1
              }
s.run_Tdependence(betas, stops, Gammas, parameters, evaluate_transport_DC=True, evaluate_vertex_DC=False)
 
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
results_x, results_y = tokovi.compute_chi(omega0, Gamma, mu_, invt, nodes, weights, s.thetas, s.parities, s.tok_x1, s.mat_x, s.tok_y1, s.mat_y, s.energije, rho_tilde_factory, rho_tilde_cache, eps=eps, n_workers=n_workers, verbose=True)

plt.plot(omega0, -results_y['chi_jj0'].imag / omega0)
plt.plot(omega0, -results_y['chi_jj0'].imag / omega0 - results_y['dchi_jj'].imag / omega0)
plt.show()