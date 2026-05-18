
import numpy as np
import sys
from scipy.special import roots_legendre
import json

sys.path.append("/Users/ana/Desktop/tns-ana/main-programs/") 

import tns_helpers as helpers
import tns_tokovi as tokovi

''' create TNS class '''
class TNS:
    def __init__(self, file, rho=None, energije=None, fs=None, vecs=None, fock=None, hartree=None):
                 
                 #a, b, b2, c, Ny, Nx, U, V, mu0, parameters1, parameters2, eps0, n_target, faktor=1.,
                 #rho=None, energije=None, fs=None, vecs=None, fock=None, hartree=None):
        
        with open(file, "r", encoding="utf-8") as f:
            params = json.load(f)
        
        U = params["U"] # eV
        V = params["V"] # eV
        a = params["a"] # A
        b = params["b"] # A
        b2 = params["b2"] # A
        c = params["c"] # A

        parameters1 = list(params["parameters1"].values())
        parameters2 = list(params["parameters2"].values())

        Ny, Nx = params["Ny"], params["Nx"]
        mu0 = params["mu"]
        eps0 = params["eps0"]

        n_target = params["n_target"]
        faktor = params["faktor"]

        Ny = params["Ny"]
        Nx = params["Nx"]

        self.a, self.b, self.b2, self.c = a, b, b2, c
        self.U, self.V = U, V
        self.faktor = faktor
        self.n_target = n_target
        self.parameters1 = parameters1
        self.parameters2 = parameters2
        self.Nx, self.Ny = Nx, Ny
        self.Nk = Ny * Nx
        Ky = 2*np.pi/b * np.arange(-Ny//2, Ny//2) / Ny
        Kx = 2*np.pi/a * np.arange(-Nx/2, Nx//2) / Nx 
        Kxmesh, Kymesh = np.meshgrid(Kx, Ky)
        self.nx = Kymesh.shape[1]
        self.kxmesh = Kxmesh
        self.kymesh = Kymesh
        self.hop = helpers.H_hopping(self.kymesh, self.kxmesh, a, b, faktor=self.faktor)
        self.perturb = helpers.H_perturb(self.kymesh, self.kxmesh, a, b)
        self.rho = helpers.Rho0(self.Ny, self.Nx)
        self.mu = mu0

        self.fock = helpers.H_fock(self.kxmesh, self.Nk, self.rho, a, V)
        self.hartree = helpers.H_hartree(self.rho, self.Nk, U, V)

        if energije == None:
            self.rho, self.energije, self.fs, self.vecs, self.err, self.n, self.fock, self.hartree = helpers.GS(self.kxmesh, self.rho, self.hop, self.perturb, self.hartree, self.fock, self.mu, eps0, a, U, V, epsilon=1e-10, maxiter=3000, N_epsilon=5)
        else:
            self.rho, self.energije, self.fs, self.vecs, self.err, self.n, self.fock, self.hartree = rho, energije, fs, vecs, 0.0, 2.0, fock, hartree
        self.mu = 0.5 * (np.min(self.energije[2]) + np.max(self.energije[1]))

        self.kinetic_extend = tokovi.kinetic(extend=True, faktor=self.faktor)
        self.kinetic = tokovi.kinetic(faktor=self.faktor)
        self.tok = tokovi.j_tok(self.kymesh, self.kxmesh, self.a, self.b, self.b2, self.kinetic)
        self.dH_dk = tokovi.velocity_HF(self.kymesh, self.kxmesh, self.a, self.b, self.kinetic)
        self.interaction = tokovi.interaction()
        self.pos = tokovi.positions(self.a, self.b, self.b2)
        self.geom, self.phases = tokovi.input_data(self.kymesh, self.kxmesh, a, b, self.pos, self.kinetic_extend, self.interaction)
        #self.g_ffts = tokovi.G_ffts(self.phases, self.Ny, self.nx)
        self.interaction_exp, self.thetas, self.parities = tokovi.interaction_expand(self.interaction, self.U, self.V, self.a)
        self.rhos = tokovi.rho_operators(len(self.thetas), self.kymesh, self.kxmesh, self.interaction, self.a, self.b)

        self.velocities()

        self.rho0 = self.rho
        self.mu0 = self.mu

        self.phi = helpers.Phi(self.kxmesh, self.rho, self.a)[0].real
        self.Phi_x = []
        self.Phi_y = []
        
        self.phis = []
        self.mus = []
        self.transport_gaps = []
        self.optical_gaps = []
        self.errors = []
        self.occupations = []

        self.Ts = []

        self.L11x_boltz = []
        self.L11y_boltz = []
        self.L12x_boltz = []
        self.L12y_boltz = []

        self.L11x = []
        self.L11y = []
        self.L12x = []
        self.L12y = []
        self.L12qx = []
        self.L12qy = []    

        self.L11x_0 = []
        self.L12x_0 = []
        self.L12qx_0 = []
        self.L11x_corr = []
        self.L12x_corr = []
        self.L12qx_corr = []

        self.L11y_0 = []
        self.L12y_0 = []
        self.L12qy_0 = []
        self.L11y_corr = []
        self.L12y_corr = []
        self.L12qy_corr = []

    def velocities(self) -> None:
        # Boltzmann's group velocities using Hellmann-Feynmann
        self.dfock_dk = tokovi.velocity_fock_HF(self.rho, self.kymesh, self.kxmesh, self.a, self.V)
        self.velocity_x = np.einsum('iixy-> ixy', tokovi.operator_tilde(self.dH_dk[0] + self.dfock_dk[0], self.vecs).real)
        self.velocity_y = np.einsum('iixy-> ixy', tokovi.operator_tilde(self.dH_dk[1] + self.dfock_dk[1], self.vecs).real)

        # Kubo's velocities: expectation values of current operator
        self.tok_x1 = tokovi.operator_tilde(self.tok[0], self.vecs)
        self.tok_y1 = tokovi.operator_tilde(self.tok[1], self.vecs)

        # non-local interaction current
        mat1, mat2, mat3, mat4 = tokovi.compute_all_mf_matrices(self.kymesh, self.rho, self.geom, self.phases, self.a, self.b, self.U, self.V)
        mat = mat1 + mat2 + mat3 + mat4
        mat_tilde = tokovi.operator_tilde(mat, self.vecs)
        self.mat_x = mat_tilde[0]
        self.mat_y = mat_tilde[1]
        
    def density_of_states(self, omegas, faktor):
        return helpers.DoS(self.kymesh, self.kxmesh, self.energije, omegas, self.mu, self.velocity_x, self.velocity_y, faktor=faktor)

    def next_T(self, T, i) -> None:
        if i == 1: dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials = self.parameters1
        elif i ==2: dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials = self.parameters2
        if len(self.mus) > 1:
            mu_candidate = self.mu
        else:
            mu_candidate = self.mu
        rho, energije, fs, vecs, fock, hartree, err, n, mu = helpers.NewMu(self.n_target, self.kxmesh, self.rho, self.hop, self.perturb, self.hartree, self.fock,
                                                                self.a, self.U, self.V, T, mu_candidate, 
                                                                dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials)
        self.rho = rho
        self.energije = energije
        self.fs = fs
        self.vecs = vecs
        self.fock = fock
        self.hartree = hartree
        self.mu = mu
        self.err = err
        self.n = n

    def run_Tdependence(self, betas, stops, Gammas, params, evaluate_transport_DC=True, evaluate_vertex_DC=True, save_during=False, file_name=None):
        Nomega = params['Nomega']
        eps = params['eps']
        omega0 = params['omega0']
        eps2 = params['eps2']
        deg = params['deg']
        n_workers = params['n_workers']

        if evaluate_vertex_DC:
            nodes, weights = roots_legendre(deg)

        for i, beta in enumerate(betas):
            T = 1/beta
            self.T = T
            if i not in stops:
                print(f'{i/len(betas)} started evaluating at beta={beta}', flush=True)
                if (i+1) in stops:
                    rho_save = self.rho
                    energije_save = self.energije
                    fs_save = self.fs
                    vecs_save = self.vecs
                    fock_save = self.fock
                    hartree_save = self.hartree
                    mu_save = self.mu
                    err_save = self.err
                    n_save = self.n
                self.next_T(T, 2)
            else:
                print(f'{i/len(betas)} started evaluating at beta={beta}', flush=True)
                self.next_T(T, 1)
                self.Ts.append(T)
                self.phis.append(helpers.Phi(self.kxmesh, self.rho, self.a)[0].real)
                print(f'----{self.phis[-1]}----')
                self.transport_gaps.append(np.min(self.energije[2]) - np.max(self.energije[1]))
                self.optical_gaps.append(np.min(self.energije[2] - self.energije[1]))
                self.mus.append(self.mu)
                self.errors.append(self.err)
                self.occupations.append(self.n)
                #print(f'occupation error is {np.abs(self.n - self.n_target)}')

                if evaluate_transport_DC:
                    self.DC_coefficients(eps, Nomega, Gammas)

                if evaluate_vertex_DC:
                    self.DC_bubble_corr(nodes, weights, Gammas, omega0, eps2, n_workers)

                if i > 0:
                    self.rho = rho_save
                    self.energije = energije_save
                    self.fs = fs_save
                    self.vecs = vecs_save
                    self.fock = fock_save
                    self.hartree = hartree_save
                    self.mu = mu_save
                    self.err = err_save
                    self.n = n_save

                if save_during:
                    results_intermediate = self.collect_results()
                    np.savez(f'{file_name}.npz', **results_intermediate)

    def ls_kubo(self, epsilons, Gamma, mfd1):
        phi_x = tokovi.phi_Kubo(self.tok_x1, self.tok_x1, epsilons, self.energije, Gamma, self.mu)
        phi_y = tokovi.phi_Kubo(self.tok_y1, self.tok_y1, epsilons, self.energije, Gamma, self.mu)
        phiQ_x = tokovi.phi_Kubo(self.mat_x, self.tok_x1, epsilons, self.energije, Gamma, self.mu)
        phiQ_y = tokovi.phi_Kubo(self.mat_y, self.tok_y1, epsilons, self.energije, Gamma, self.mu)

        l11_x = np.pi * tokovi.integral_omega(phi_x * mfd1, epsilons)
        l12_x = np.pi * tokovi.integral_omega(epsilons * phi_x * mfd1, epsilons)
        l12q_x = np.pi * tokovi.integral_omega(phiQ_x * mfd1, epsilons)

        l11_y = np.pi * tokovi.integral_omega(phi_y * mfd1, epsilons)
        l12_y = np.pi * tokovi.integral_omega(epsilons * phi_y * mfd1, epsilons)
        l12q_y = np.pi * tokovi.integral_omega(phiQ_y * mfd1, epsilons)

        return l11_x, l12_x, l12q_x, l11_y, l12_y, l12q_y
    
    def DC_coefficients(self, eps, Nomega, Gammas):
        T = self.Ts[-1]
        epsilon_max = np.sqrt(np.abs(np.arccosh(1/(eps*4*T))) * 2 * T)
        epsilons = np.linspace(-epsilon_max, epsilon_max, Nomega, dtype=np.float64)
        mfd1 = -tokovi.fd_1(epsilons, T)

        Ngamma = len(Gammas)
        self.velocities()

        # Boltzmann coefficients
        l11x_boltz = np.zeros(Ngamma)
        l11y_boltz = np.zeros(Ngamma)
        l12x_boltz = np.zeros(Ngamma)
        l12y_boltz = np.zeros(Ngamma)
        K0b_x, K0b_y, K1b_x, K1b_y = tokovi.Kn_boltzmann(self.velocity_x, self.velocity_y, self.energije, self.mu, T)

        # Kubo coefficients
        l11x = np.zeros(Ngamma)
        l11y = np.zeros(Ngamma)
        l12x = np.zeros(Ngamma)
        l12y = np.zeros(Ngamma)
        l12qx = np.zeros(Ngamma)
        l12qy = np.zeros(Ngamma)

        for g, Gamma in enumerate(Gammas):
            l11x_boltz[g] = K0b_x / (2*Gamma)
            l11y_boltz[g] = K0b_y / (2*Gamma)
            l12x_boltz[g] = K1b_x / (2*Gamma)
            l12y_boltz[g] = K1b_y / (2*Gamma)

            l11_x, l12_x, l12q_x, l11_y, l12_y, l12q_y = self.ls_kubo(epsilons, Gamma, mfd1)
            l11x[g] = l11_x.real
            l11y[g] = l11_y.real
            l12x[g] = l12_x.real
            l12y[g] = l12_y.real
            l12qx[g] = l12q_x.real
            l12qy[g] = l12q_y.real

        self.L11x_boltz.append(tokovi.to_scalar_if_single(l11x_boltz))
        self.L11y_boltz.append(tokovi.to_scalar_if_single(l11y_boltz))
        self.L12x_boltz.append(tokovi.to_scalar_if_single(l12x_boltz))
        self.L12y_boltz.append(tokovi.to_scalar_if_single(l12y_boltz))
        
        self.L11x.append(tokovi.to_scalar_if_single(l11x))
        self.L11y.append(tokovi.to_scalar_if_single(l11y))
        self.L12x.append(tokovi.to_scalar_if_single(l12x))
        self.L12y.append(tokovi.to_scalar_if_single(l12y))
        self.L12qx.append(tokovi.to_scalar_if_single(l12qx))
        self.L12qy.append(tokovi.to_scalar_if_single(l12qy))

    def DC_bubble_corr(self, nodes, weights, Gammas, omega0, eps, n_workers=None):
        Ngamma = len(Gammas)

        l11x_0 = np.zeros(Ngamma)
        l12x_0 = np.zeros_like(l11x_0)
        l12qx_0 = np.zeros_like(l11x_0)

        l11x = np.zeros_like(l11x_0)
        l12x = np.zeros_like(l12x_0)
        l12qx = np.zeros_like(l12qx_0)

        l11y_0 = np.zeros_like(l11x_0)
        l12y_0 = np.zeros_like(l11x_0)
        l12qy_0 = np.zeros_like(l11x_0)

        l11y = np.zeros_like(l11x_0)
        l12y = np.zeros_like(l12x_0)
        l12qy = np.zeros_like(l12qx_0)

        self.factory = tokovi.make_rho_tilde_factory(self.interaction, self.a, self.b, self.kymesh, self.kxmesh, self.vecs)

        for g, Gamma in enumerate(Gammas):
            mu_ = self.mu / Gamma
            invt = Gamma / self.Ts[-1]

            rho_tilde_cache = {}
            rho_tilde_factory = tokovi.make_rho_tilde_factory(self.interaction, self.a, self.b, self.kymesh, self.kxmesh, self.vecs)
            results_x, results_y = tokovi.compute_chi(omega0, Gamma, mu_, invt, nodes, weights, self.thetas, self.parities, self.tok_x1, self.mat_x, self.tok_y1, self.mat_y, self.energije, rho_tilde_factory, rho_tilde_cache, eps=eps, n_workers=n_workers, verbose=True)

            Chi_jj0 = - results_x['chi_jj0'].imag
            dChi_jj  = - results_x['dchi_jj'].imag
            Chi_jj = Chi_jj0 + dChi_jj
            left, right = tokovi.find_flat_regime(omega0, Chi_jj0)
            l11x_0[g] = tokovi.get_dc_coefficient(omega0[left:right], Chi_jj0[left:right])[0]
            left, right = tokovi.find_flat_regime(omega0, Chi_jj)
            l11x[g] = tokovi.get_dc_coefficient(omega0[left:right], Chi_jj[left:right])[0]

            Chi_jEj0 = - results_x['chi_jEj0'].imag
            dChi_jEj = - results_x['dchi_jEj'].imag
            Chi_jEj = Chi_jEj0 + dChi_jEj
            left, right = tokovi.find_flat_regime(omega0, Chi_jEj0)
            l12x_0[g] = tokovi.get_dc_coefficient(omega0[left:right], Chi_jEj0[left:right])[0]
            left, right = tokovi.find_flat_regime(omega0, Chi_jEj)
            l12x[g] = tokovi.get_dc_coefficient(omega0[left:right], Chi_jEj[left:right])[0]

            Chi_matj0 = - results_x['chi_matj0'].imag
            dChi_matj = - results_x['dchi_matj'].imag
            Chi_matj = Chi_matj0 + dChi_matj
            left, right = tokovi.find_flat_regime(omega0, Chi_matj0)
            l12qx_0[g] = tokovi.get_dc_coefficient(omega0[left:right], Chi_matj0[left:right])[0]
            left, right = tokovi.find_flat_regime(omega0, Chi_matj)
            l12qx[g] = tokovi.get_dc_coefficient(omega0[left:right], Chi_matj[left:right])[0]

            Chi_jj0 = - results_y['chi_jj0'].imag
            dChi_jj  = - results_y['dchi_jj'].imag
            Chi_jj = Chi_jj0 + dChi_jj
            left, right = tokovi.find_flat_regime(omega0, Chi_jj0)
            l11y_0[g] = tokovi.get_dc_coefficient(omega0[left:right], Chi_jj0[left:right])[0]
            left, right = tokovi.find_flat_regime(omega0, Chi_jj)
            l11y[g] = tokovi.get_dc_coefficient(omega0[left:right], Chi_jj[left:right])[0]

            Chi_jEj0 = - results_y['chi_jEj0'].imag
            dChi_jEj = - results_y['dchi_jEj'].imag
            Chi_jEj = Chi_jEj0 + dChi_jEj
            left, right = tokovi.find_flat_regime(omega0, Chi_jEj0)
            l12y_0[g] = tokovi.get_dc_coefficient(omega0[left:right], Chi_jEj0[left:right])[0]
            left, right = tokovi.find_flat_regime(omega0, Chi_jEj)
            l12y[g] = tokovi.get_dc_coefficient(omega0[left:right], Chi_jEj[left:right])[0]

            Chi_matj0 = - results_y['chi_matj0'].imag
            dChi_matj = - results_y['dchi_matj'].imag
            Chi_matj = Chi_matj0 + dChi_matj
            left, right = tokovi.find_flat_regime(omega0, Chi_matj0)
            l12qy_0[g] = tokovi.get_dc_coefficient(omega0[left:right], Chi_matj0[left:right])[0]
            left, right = tokovi.find_flat_regime(omega0, Chi_matj)
            l12qy[g] = tokovi.get_dc_coefficient(omega0[left:right], Chi_matj[left:right])[0]

        self.L11x_0.append(tokovi.to_scalar_if_single(l11x_0))
        self.L12x_0.append(tokovi.to_scalar_if_single(l12x_0))
        self.L12qx_0.append(tokovi.to_scalar_if_single(l12qx_0))
        self.L11x_corr.append(tokovi.to_scalar_if_single(l11x))
        self.L12x_corr.append(tokovi.to_scalar_if_single(l12x))
        self.L12qx_corr.append(tokovi.to_scalar_if_single(l12qx))
        self.L11y_0.append(tokovi.to_scalar_if_single(l11y_0))
        self.L12y_0.append(tokovi.to_scalar_if_single(l12y_0))
        self.L12qy_0.append(tokovi.to_scalar_if_single(l12qy_0))
        self.L11y_corr.append(tokovi.to_scalar_if_single(l11y))
        self.L12y_corr.append(tokovi.to_scalar_if_single(l12y))
        self.L12qy_corr.append(tokovi.to_scalar_if_single(l12qy))

    def reset(self):
        self.rho = self.rho0
        self.hartree = helpers.H_hartree(self.rho, self.Nk, self.U, self.V)
        self.fock = helpers.H_fock(self.kxmesh, self.Nk, self.rho, self.a, self.V)
        self.mu = self.mu0

    def reset_infty(self):
        self.rho = helpers.Rhoinfty(self.Ny, self.Nx)
        self.hartree = helpers.H_hartree(self.rho, self.Nk, self.U, self.V)
        self.fock = helpers.H_fock(self.kxmesh, self.Nk, self.rho, self.a, self.V)
        
        _, energije, fs, vecs, _, _, _, _ = helpers.Rho_next(self.kxmesh, self.kymesh, self.rho, self.hop, self.perturb, self.hartree, self.fock, self.a, self.b, self.b2, self.U, self.V, 0, self.mu, 50, 0.5, 1e-10, eps0=0.0, N_epsilon=5)
        self.energije = energije
        self.fs = fs
        self.vecs = vecs

    def save_data(self):
        data = {'rho' : self.rho,
                'energije' : self.energije,
                'fs' : self.fs,
                'vecs' : self.vecs,
                'fock' : self.fock,
                'hartree' : self.hartree,
                'mu' : self.mu,
                'phi' : self.phi
                }
        return data
    
    # remember state with n=2 at some temperature
    def remember_T(self) -> None:
        self.mu_T = self.mu
        self.rho_T = self.rho
        self.energije_T = self.energije
        self.vecs_T = self.vecs

    # go back to state with n=2 at some temperature
    def revisit_T(self) -> None:
        self.mu = self.mu_T
        self.rho = self.rho_T
        self.hartree = helpers.H_hartree(self.rho, self.Nk, self.U, self.V)
        self.fock = helpers.H_fock(self.kxmesh, self.Nk, self.rho, self.a, self.V)
        self.energije = self.energije_T
        self.vecs = self.vecs_T

    def newOccupation(self, n_target, Dmu, steps, epsilon, maxiter, mix=0.5):
        n0 = helpers.Occupation(self.rho)
        Dmu = Dmu*np.sign(n_target - n0)
        dmu = Dmu / steps
        rho = self.rho
        hartree = self.hartree
        fock = self.fock
        mu = self.mu
        for i in range(steps):
            rho_new, energije_new, _, vecs_new, fock_new, hartree_new, err, n = helpers.Rho_next(self.kxmesh, rho, self.hop, self.perturb, hartree, fock, self.a, self.U, self.V, self.T, mu + i*dmu, maxiter, mix, epsilon, 0.0, 5)
            if np.sign(n_target - n0) * np.sign(n - n_target) == +1:
                break
            print(n, flush=True)
        mu = mu + i*dmu
        self.rho = rho_new
        self.energije = energije_new
        self.vecs = vecs_new
        self.hartree = hartree_new
        self.fock = fock_new
        self.mu = mu
        return n, mu, err

    def converge_newOccupation(self, n_target, Dmu, steps, epsilon, maxiter, n_pass, mix=0.5):
        """
        1. Adiabatically walk from current n (~2) toward n_target using fixed Dmu steps
        2. Once close enough, refine with bisection
        """

        n_current = helpers.Occupation(self.rho)
        print(f"Starting occupation: n={n_current:.6f}, target: n_target={n_target:.6f}", flush=True)

        # ---- Phase 1: Adiabatic walk ----
        # keep stepping with Dmu until we overshoot or get close
        step_count = 0

        err = 0.0
        while np.abs(n_current - n_target) > n_pass:
            mu_before = self.mu

            n_new, _, err = self.newOccupation(n_target, Dmu, steps, epsilon, maxiter, mix)
            step_count += 1
            print(f"Walk step {step_count}: n={n_new:.6f}, mu={self.mu:.6f}", flush=True)

            # check if we overshot
            if np.sign(n_new - n_target) != np.sign(n_current - n_target):
                print("Overshot! Entering bisection...", flush=True)
                mu_lo = min(mu_before, self.mu)
                mu_hi = max(mu_before, self.mu)
                break

            n_current = n_new

            if np.abs(n_current - n_target) < n_pass:
                print(f"Converged during walk: n={n_current:.6f}", flush=True)
                return n_current, self.mu, err

        else:
            # exited while loop without overshoot — already converged
            return n_current, self.mu, err

        # ---- Phase 2: Bisection refinement ----
        for i in range(50):
            mu_mid = (mu_lo + mu_hi) / 2.0

            self.mu = mu_mid

            n_mid, _, err = self.newOccupation(n_target, 0.0, 1, epsilon, maxiter, mix)

            print(f"Bisect {i+1}: mu={mu_mid:.6f}, n={n_mid:.6f}, |dn|={abs(n_mid - n_target):.2e}", flush=True)

            if np.abs(n_mid - n_target) < n_pass:
                print(f"Converged in {i+1} bisection steps.", flush=True)
                return n_mid, mu_mid, err

            if n_mid < n_target:
                mu_lo = mu_mid
            else:
                mu_hi = mu_mid

        print("Warning: bisection did not fully converge.", flush=True)
        return n_mid, mu_mid, err

    def collect_results(self):
        results = {
            "phis": self.phis,
            "mus": self.mus,
            "transport_gaps": self.transport_gaps,
            "optical_gaps": self.optical_gaps,
            "errors": self.errors,
            "occupations": self.occupations,

            "Ts": self.Ts,

            "L11x_boltz" : self.L11x_boltz,
            "L11y_boltz" : self.L11y_boltz,
            "L12x_boltz" : self.L12x_boltz,
            "L12y_boltz" : self.L12y_boltz,

            "L11x" : self.L11x,
            "L11y" : self.L11y,
            "L12x" : self.L12x,
            "L12y" : self.L12y,
            "L12qx" : self.L12qx,
            "L12qy" : self.L12qy,

            "L11x_0" : self.L11x_0,
            "L12x_0" : self.L12x_0,
            "L12qx_0" : self.L12qx_0,
            "L11x_corr" : self.L11x_corr,
            "L12x_corr" : self.L12x_corr,
            "L12qx_corr" : self.L12qx_corr,

            "L11y_0" : self.L11y_0,
            "L12y_0" : self.L12y_0,
            "L12qy_0" : self.L12qy_0,
            "L11y_corr" : self.L11y_corr,
            "L12y_corr" : self.L12y_corr,
            "L12qy_corr" : self.L12qy_corr
            }
        return results