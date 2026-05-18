import os, sys
os.environ['PATH'] = '/Library/TeX/texbin:' + os.environ['PATH']
sys.path.append("/Users/ana/Desktop/tns-ana/main-programs/")
import numpy as np
from scipy.special import roots_legendre
import matplotlib
import matplotlib.pyplot as plt

import os
from tns_module import * 
from tns_helpers import *
import tns_tokovi as tokovi

from scipy.optimize import curve_fit
import seaborn as sns

U = 2.5 # eV
V = 0.785 # eV
a = 3.503 #3.51 # A
b = 15.761 #15.79 # A
b2 = 1.927 # A
c = 13.42 # A
params = {
    "a": 3.51,
    "b": 15.79,
    "b2": 1.927,
    "c": 13.42,
    "U": 2.5,
    "V": 0.785,
    "Ny": 80,
    "Nx": 80,
    "mu": 2.84,
    "eps0": 0.1,
    "n_target": 2.0,
    "faktor": 1.0,
    "beta0": 250,
    "scale": 1.005,
    "Nbetas": 20,
    "freq_betas": 3,
    "eps": 1e-05,
    "Nomega": 501,
    "Gammas": [
        0.008
    ],
    "parameters1": {
        "dmu": 0.001,
        "maxiter": 200,
        "maxiter_last": 1000,
        "eps_last": 1e-07,
        "mix": 0.5,
        "mix2": 0.005,
        "mix3": 1.5,
        "n_pass": 1e-06,
        "max_trials": 10
    },
    "parameters2": {
        "dmu": 0.001,
        "maxiter": 50,
        "maxiter_last": 100,
        "eps_last": 1e-07,
        "mix": 0.5,
        "mix2": 0.005,
        "mix3": 1.5,
        "n_pass": 1e-04,
        "max_trials": 10
    },
    "constants": {
        "inv_G0": 25.81280745,
        "kb": 8.6173303
    }}

U = params["U"] # eV
V = params["V"] # eV
a = params["a"] # A
b = params["b"] # A
b2 = params["b2"] # A
c = params["c"] # A

inv_G0 = params["constants"]["inv_G0"] * 1e3 # Ohm
kb = params["constants"]["kb"] * 1e-5 # eV/K
constants = [inv_G0, kb]

parameters1 = list(params["parameters1"].values())
parameters2 = list(params["parameters2"].values())

Ny, Nx = params["Ny"], params["Nx"]
mu = params["mu"]
eps0 = params["eps0"]

n_target = params["n_target"]
faktor = 1.0 #params["faktor"]

n_target = 2.0
faktor = 1.0

Ny,Nx=80,80
#gs_data = np.load("/Users/ana/Desktop/ta2nise5-new/tns_data/data_0_250.npz")
s = TNS(a, b, b2, c, Ny, Nx, U, V, mu, parameters1, parameters1, eps0, n_target, faktor=faktor,)

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

plt.plot(omega0, -results_x['chi_jj0'].imag / omega0)
plt.plot(omega0, -results_x['chi_jj0'].imag / omega0 - results_x['dchi_jj'].imag / omega0)
plt.show()