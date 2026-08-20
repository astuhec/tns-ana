
import numpy as np
import sys
from scipy.special import roots_legendre
import json

sys.path.append("/home/stuhecana/tns-ana/main-programs/") 

import tns_helpers as helpers
import tns_tokovi as tokovi

''' create TNS class '''
class TNS:
    def __init__(self, input_file, hopping_file, interaction_file, perturbation_file,
                 Ny=None, Nx=None, rho=None, energije=None, fs=None, vecs=None, fock=None, hartree=None, pos=None, faktor=None, V=None, U=None, mu=None):
        
        ''' read input parameter and initialize the system '''
        with open(input_file, "r", encoding="utf-8") as f:
            params = json.load(f)

        self.Ny = params["Ny"] if Ny==None else Ny
        self.Nx = params["Nx"] if Nx==None else Nx

        self.N_epsilon = params["N_epsilon"]
        
        print(f'=' * 80 + '\n' + 'Started TNS calculation' + '\n' + f'=' * 80, flush=True)
        print(f'Initialized lattice with Ny={self.Ny} and Nx={self.Nx} unit cells.', flush=True)
        self.Nk = self.Ny * self.Nx
        self.U = params["U"] if U==None else U # eV
        self.V = params["V"] if V==None else V # eV
        self.a = params["a"] # A
        self.b = params["b"] # A
        self.b2 = params["b2"] # A
        self.c = params["c"] # A

        self.parameters1 = list(params["parameters1"].values())
        self.parameters2 = list(params["parameters2"].values())

        self.mu = params["mu"]
        eps0 = params["eps0"]

        self.n_target = params["n_target"]
        self.faktor = params["faktor"] if faktor==None else faktor

        Ky = 2*np.pi/self.b * np.arange(-self.Ny//2, self.Ny//2) / self.Ny
        Kx = 2*np.pi/self.a * np.arange(-self.Nx//2, self.Nx//2) / self.Nx 
        Kxmesh, Kymesh = np.meshgrid(Kx, Ky)
        self.nx = Kymesh.shape[1]
        self.kxmesh = Kxmesh
        self.kymesh = Kymesh

        self.hop = helpers.H_hopping(self.kymesh, self.kxmesh, self.a, self.b, faktor=self.faktor, file=hopping_file)
        self.perturb = helpers.H_perturb(self.kymesh, self.kxmesh, self.a, self.b, file=perturbation_file)
        self.hartree_list = helpers.make_hartree_list(interaction_file)

        self.rho = helpers.Rho0(self.Ny, self.Nx)
        self.fock = helpers.H_fock(self.kxmesh, self.Nk, self.rho, self.a, self.V)
        self.hartree = helpers.H_hartree(self.rho, self.Nk, self.U, self.V, self.hartree_list)

        ''' if ground state is not provided, compute it 
            else: use the provided gorund state'''
        if mu==None:
            self.rho, self.energije, self.fs, self.vecs, self.fock, self.hartree, self.err, self.n = helpers.GS(self.kxmesh, self.rho, self.hop, self.perturb, self.hartree, self.fock, self.mu, eps0, self.a, self.U, self.V, epsilon=1e-12, maxiter=10000, N_epsilon=5, hartree_list=self.hartree_list)
            self.mu = 0.5 * (np.min(self.energije[2]) + np.max(self.energije[1]))
        else:
            self.rho, self.energije, self.fs, self.vecs, self.err, self.n, self.fock, self.hartree = rho, energije, fs, vecs, 0.0, 2.0, fock, hartree
            self.mu = mu
        self.rho0 = self.rho
        self.mu0 = self.mu

        phi_full = helpers.Phi(self.kxmesh, self.rho, self.a).real
        self.phi = phi_full[0]
        pos12 = np.array([0.,self.a])
        pos34 = np.array([0.,-self.a])
        print('-' * 80, flush=True)
        print(f'Found ground state.' + '\n' + 'Order parameter components are:' + '\n' + \
              f'phi_15 = {np.round(phi_full[0], 8)}' + '\n' + \
              f'phi_25 = {np.round(phi_full[1], 8)}' + '\n' + \
              f'phi_36 = {np.round(phi_full[2], 8)}' + '\n' + \
              f'phi_36 = {np.round(phi_full[3], 8)}' + '\n' + '-' * 80, flush=True)
        print('Delta components are:' + '\n' + 
              f'Deltas_15: {helpers.Delta_full(self.kxmesh, self.Nk, self.rho, 0, 4, pos12).real}' + '\n' + 
              f'Deltas_25: {helpers.Delta_full(self.kxmesh, self.Nk, self.rho, 1, 4, pos12).real}' + '\n' + 
              f'Deltas_36: {helpers.Delta_full(self.kxmesh, self.Nk, self.rho, 2, 5, pos34).real}' + '\n' + 
              f'Deltas_46: {helpers.Delta_full(self.kxmesh, self.Nk, self.rho, 3, 5, pos34).real}' + '\n'  + '-' * 80 + '\n' + 
              f'Chemical potential is {np.round(self.mu, 5)} eV' + '\n' + \
              f'Occupation is {np.round(self.n, 5)}' + '\n' + '-' * 80, flush=True
            )
        
        self.kinetic_extend = tokovi.kinetic(extend=True, faktor=self.faktor, file=hopping_file)
        self.kinetic = tokovi.kinetic(faktor=self.faktor, file=hopping_file)
        # self.kinetic has already been rescaled by self.faktor in tokovi.kinetic.
        self.tok = tokovi.j_tok(self.kymesh, self.kxmesh, self.a, self.b, self.b2, self.kinetic, pos=pos)
        self.dH_dk = tokovi.velocity_HF(self.kymesh, self.kxmesh, self.a, self.b, self.kinetic)
        self.interaction = tokovi.interaction(file=interaction_file)
        if pos==None:
            self.pos = tokovi.positions(self.a, self.b, self.b2)
        elif pos==0:
            pos = []
            for i in range(7):
                pos.append([0.0,0.0])
            self.pos = np.array(pos)
        self.geom, self.phases = tokovi.input_data(self.kymesh, self.kxmesh, self.a, self.b, self.pos, self.kinetic_extend, self.interaction)
        self.thetas = tokovi.thetas_kernel(self.interaction, self.U, self.V, self.a)
        self.velocities()
        
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

        # Kubo's velocities
        self.current_x = tokovi.hermitize_operator(tokovi.operator_tilde(self.tok[0], self.vecs))
        self.current_y = tokovi.hermitize_operator(tokovi.operator_tilde(self.tok[1], self.vecs))

        # non-local interaction current
        mat1, mat2, mat3, mat4 = tokovi.compute_all_mf_matrices(self.kymesh, self.rho, self.geom, self.phases, self.a, self.b, self.U, self.V)
        mat = mat1 + mat2 + mat3 + mat4
        mat_dag = np.swapaxes(mat.conj(), 1, 2)
        relative_error = np.linalg.norm(mat - mat_dag) / max(np.linalg.norm(mat), 1e-30)
        assert relative_error < 1e-10, (
            f"Nonlocal MF current is not Hermitian: {relative_error}"
        )
        mat_tilde = tokovi.hermitize_operator(tokovi.operator_tilde(mat, self.vecs))
        self.mat_x = mat_tilde[0]
        self.mat_y = mat_tilde[1]
        
    def density_of_states(self, epsilons):
        return helpers.DoS(self.kymesh, self.kxmesh, self.energije, epsilons, self.mu, self.velocity_x, self.velocity_y, faktor=self.faktor)

    def next_T(self, i) -> None:
        if i == 1:
            dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials = self.parameters1
        elif i == 2:
            dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials = self.parameters2

        mu_candidate = self.mu
        rho, energije, fs, vecs, fock, hartree, err, n, mu = helpers.NewMu(self.n_target, self.kxmesh, self.rho, self.hop, self.perturb, self.hartree, self.fock,
                                                                self.a, self.U, self.V, self.T, mu_candidate, 
                                                                dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials, hartree_list=self.hartree_list)
        self.rho = rho
        self.energije = energije
        self.fs = fs
        self.vecs = vecs
        self.fock = fock
        self.hartree = hartree
        self.mu = mu
        self.err = err
        self.n = n

    def run_Tdependence(self, input_temperature, save_during=False, file_name=None, own_beta=None, betas_own=None, stops_own=None):
        
        with open(input_temperature, "r", encoding="utf-8") as f:
            params_all = json.load(f)

        evaluate_transport_DC = params_all['evaluate_transport_DC']
        evaluate_vertex_DC = params_all['evaluate_vertex_DC']

        print('Started to find temperature dependence of transport coefficients.', flush=True)
        if evaluate_transport_DC == True:
            print('Will calculate Boltzmann and Kubo bubble DC coefficients.', flush=True)
        if evaluate_vertex_DC == True:
            print('Will calculate Kubo bubble DC coefficients and vertex corrections.', flush=True)
        if evaluate_transport_DC == False and evaluate_vertex_DC == False:
            print('Will not calculate transport coefficients, but will find self-consistent rho(T) and mu(T).', flush=True)
        Gammas = params_all['Gammas']
        params = params_all['params']
        Nomega = params['Nomega']
        eps = params['eps']
        omega0_low = params['omega0_low']
        omega0_high = params['omega0_high']
        omega0_len = params['omega0_len']
        omega0 = np.logspace(omega0_low, omega0_high, omega0_len)
        eps2 = params['eps2']
        deg = params['deg']
        n_workers = params['n_workers']

        betas0 = params_all['beta0']
        scale = params_all['scale']
        Nbetas = params_all['Nbetas']
        freq_betas = params_all['freq_betas']

        if own_beta==None:
            betas = betas0 / scale**np.arange(1, Nbetas+1)
            max_stop = int(Nbetas // freq_betas)
            stops = [freq_betas*i for i in range(1, max_stop+1)]
        else:
            betas = betas_own
            stops = stops_own

        if evaluate_vertex_DC:
            nodes, weights = roots_legendre(deg)

        for i, beta in enumerate(betas):
            T = 1/beta
            self.T = T
            if i not in stops:
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
                self.next_T(2)
            else:
                self.next_T(1)
                self.Ts.append(T)
                self.phis.append(helpers.Phi(self.kxmesh, self.rho, self.a)[0].real)
                print(f'Progress {i/len(betas)}, beta={int(beta)}, phi={np.round(self.phis[-1],5)}', flush=True)
                self.transport_gaps.append(np.min(self.energije[2]) - np.max(self.energije[1]))
                self.optical_gaps.append(np.min(self.energije[2] - self.energije[1]))
                self.mus.append(self.mu)
                self.errors.append(self.err)
                self.occupations.append(self.n)

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
                    np.savez(file_name, **results_intermediate)
        print('-' * 80 + '\n' + \
              'Finished calculation.', flush=True)

    def ls_kubo(self, epsilons, Gamma, mfd1):
        phi_x = tokovi.phi_Kubo(self.current_x, self.current_x, epsilons, self.energije, Gamma, self.mu)
        phi_y = tokovi.phi_Kubo(self.current_y, self.current_y, epsilons, self.energije, Gamma, self.mu)
        phiQ_x = tokovi.phi_Kubo(self.mat_x, self.current_x, epsilons, self.energije, Gamma, self.mu)
        phiQ_y = tokovi.phi_Kubo(self.mat_y, self.current_y, epsilons, self.energije, Gamma, self.mu)

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
        self.velocities()
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

            rho_tilde_factory = tokovi.make_rho_tilde_factory(self.interaction, self.a, self.b, self.kymesh, self.kxmesh, self.vecs)
            results_x, results_y, _, _ = tokovi.compute_chi(omega0, Gamma, mu_, invt, nodes, weights, self.thetas, self.current_x, self.mat_x, self.current_y, self.mat_y, self.energije, rho_tilde_factory, eps=eps, n_workers=n_workers, verbose=True)

            Chi_jj0 = - results_x['chi_jj0'].imag
            dChi_jj  = - results_x['dchi_jj'].imag
            Chi_jj = Chi_jj0 + dChi_jj
            l11x_0[g] = tokovi.find_DC_limit(omega0, Chi_jj0)
            l11x[g] = tokovi.find_DC_limit(omega0, Chi_jj)

            Chi_jEj0 = - results_x['chi_jEj0'].imag
            dChi_jEj = - results_x['dchi_jEj'].imag
            Chi_jEj = Chi_jEj0 + dChi_jEj
            l12x_0[g] = tokovi.find_DC_limit(omega0, Chi_jEj0)
            l12x[g] = tokovi.find_DC_limit(omega0, Chi_jEj)

            Chi_matj0 = - results_x['chi_matj0'].imag
            dChi_matj = - results_x['dchi_matj'].imag
            Chi_matj = Chi_matj0 + dChi_matj
            l12qx_0[g] = tokovi.find_DC_limit(omega0, Chi_matj0)
            l12qx[g] = tokovi.find_DC_limit(omega0, Chi_matj)

            Chi_jj0 = - results_y['chi_jj0'].imag
            dChi_jj  = - results_y['dchi_jj'].imag
            Chi_jj = Chi_jj0 + dChi_jj
            l11y_0[g] = tokovi.find_DC_limit(omega0, Chi_jj0)
            l11y[g] = tokovi.find_DC_limit(omega0, Chi_jj)

            Chi_jEj0 = - results_y['chi_jEj0'].imag
            dChi_jEj = - results_y['dchi_jEj'].imag
            Chi_jEj = Chi_jEj0 + dChi_jEj
            l12y_0[g] = tokovi.find_DC_limit(omega0, Chi_jEj0)
            l12y[g] = tokovi.find_DC_limit(omega0, Chi_jEj)

            Chi_matj0 = - results_y['chi_matj0'].imag
            dChi_matj = - results_y['dchi_matj'].imag
            Chi_matj = Chi_matj0 + dChi_matj
            l12qy_0[g] = tokovi.find_DC_limit(omega0, Chi_matj0)
            l12qy[g] = tokovi.find_DC_limit(omega0, Chi_matj)

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

    def optical_responses(self, input_optical, json_file=True):
        if json_file:
            with open(input_optical, "r", encoding="utf-8") as f:
                params = json.load(f)
        else:
            params = input_optical
        omega0_low = params['omega0_low']
        omega0_high = params['omega0_high']
        omega0_len = params['omega0_len']
        space = params['space']
        if space == 'log':
            omega0 = np.logspace(omega0_low, omega0_high, omega0_len)
        if space == 'lin':
            omega0 = np.linspace(omega0_low, omega0_high, omega0_len)

        deg = params['deg']
        nodes, weights = roots_legendre(deg)
        n_workers = params['n_workers']
        Gamma = params['Gamma']
        eps = params['eps']

        invt = Gamma / self.Ts[-1]
        mu_ = self.mu / Gamma

        rho_tilde_factory = tokovi.make_rho_tilde_factory(self.interaction, self.a, self.b, self.kymesh, self.kxmesh, self.vecs)

        results_x, results_y, results_xy, results_yx = tokovi.compute_chi(omega0, Gamma, mu_, invt, nodes, weights, self.thetas,
                                                  self.current_x, self.mat_x, self.current_y, self.mat_y,
                                                  self.energije, rho_tilde_factory,
                                                  n_workers=n_workers, eps=eps, )
        return omega0, results_x, results_y, results_xy, results_yx

    def reset(self):
        self.rho = self.rho0
        self.hartree = helpers.H_hartree(self.rho, self.Nk, self.U, self.V, self.hartree_list)
        self.fock = helpers.H_fock(self.kxmesh, self.Nk, self.rho, self.a, self.V)
        self.mu = self.mu0

    def reset_infty(self):
        self.rho = helpers.Rhoinfty(self.Ny, self.Nx)
        self.hartree = helpers.H_hartree(self.rho, self.Nk, self.U, self.V, self.hartree_list)
        self.fock = helpers.H_fock(self.kxmesh, self.Nk, self.rho, self.a, self.V)
        
        _, energije, fs, vecs, _, _, _, _ = helpers.Rho_next(self.kxmesh, self.kymesh, self.rho, self.hop, self.perturb, self.hartree, self.fock, self.a, self.b, self.b2, self.U, self.V, 0, self.mu, 50, 0.5, 1e-10, eps0=0.0, N_epsilon=5, hartree_list=self.hartree_list)
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
        self.hartree = helpers.H_hartree(self.rho, self.Nk, self.U, self.V, self.hartree_list)
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
            rho_new, energije_new, _, vecs_new, fock_new, hartree_new, err, n = helpers.Rho_next(self.kxmesh, rho, self.hop, self.perturb, hartree, fock, self.a, self.U, self.V, self.T, mu + i*dmu, maxiter, mix, epsilon, 0.0, 5, hartree_list=self.hartree_list)
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