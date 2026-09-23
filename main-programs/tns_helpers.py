import numpy as np
from scipy.special import expit
from numba import njit, prange
import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
from scipy.special import digamma

def Rho0(Ny, Nx):
    rho0 = np.zeros((6, 6, Ny, Nx), dtype='complex')
    rho0[4,4,:,:] = 1.0
    rho0[5,5,:,:] = 1.0
    return rho0

def Rhoinfty(Ny, Nx):
    rho0 = np.zeros((6, 6, Ny, Nx), dtype='complex')
    for i in range(6):
        rho0[i, i, :, :] = 2/6
    return rho0

def positions(a, b, b2):
    return np.array([[0.,0.],
                     [-a/4, b/2 - b2],
                    [-a/4,b2],
                    [a/4,-b2],
                    [a/4, -b/2 + b2],
                    [a/4,b/4],
                    [-a/4, -b/4]])

def H_hopping(Kymesh, Kxmesh, a, b, file='parametri-kinetic.txt', faktor=1.0):
    Ny, Nx = Kymesh.shape
    hop = np.zeros((6, 6, Ny, Nx), dtype='complex')
    with open(file, 'r') as f:
        for line in f:
            [x, y, orb1, orb2, t] = list(map(float, line.split()))
            orb1, orb2 = int(orb1), int(orb2)
            if orb1 != orb2: t = faktor * t
            ad = t * np.exp(-1j*(Kxmesh * x * a + Kymesh * y * b))
            hop[orb1 - 1, orb2 - 1] += ad
            if orb1 != orb2: hop[orb2 - 1, orb1 - 1] += ad.conjugate()
    return hop

def H_perturb(Kymesh, kxmesh, a, b, file='perturbacija.txt'):
    return H_hopping(Kymesh, kxmesh, a, b, file=file, faktor=1)

def Delta_full(Kxmesh, Nk, rho, i, j, x): 
    if type(x) == np.ndarray:
        return np.array([np.sum(rho[i, j] * np.exp(1j * Kxmesh * x1)) for x1 in x]) / Nk
    else: return np.sum(rho[i, j] * np.exp(1j * Kxmesh * x)) / Nk

def Phi(Kxmesh, rho, a):
    Nk = np.prod(rho.shape[-2:])
    pos12 = np.array([0.,a])
    pos34 = np.array([0.,-a])
    phi15 = np.sum(Delta_full(Kxmesh, Nk, rho, 0, 4, pos12))
    phi25 = np.sum(Delta_full(Kxmesh, Nk, rho, 1, 4, pos12))
    phi36 = np.sum(Delta_full(Kxmesh, Nk, rho, 2, 5, pos34))
    phi46 = np.sum(Delta_full(Kxmesh, Nk, rho, 3, 5, pos34))
    return np.array([phi15, phi25, phi36, phi46])

def hartree_sum(rho, orb):
    hartree = np.sum(rho[orb,orb]).real    
    return hartree

def make_hartree_list(interaction_file):
    hartree_list = []
    with open(interaction_file) as f:
        for line in f:
            [orb1, orb2] = list(map(float, line.split()))[-2:]
            orb1, orb2 = int(orb1-1), int(orb2-1)
            hartree_list.append([orb1,orb2])
    return hartree_list

def H_hartree(rho, Nk, U, V, hartree_list):
    hartree_k = np.zeros((6,6), dtype='complex')
    for line in hartree_list:
        [orb1, orb2] = line
        if orb1 == orb2: 
            hartree_k[orb1, orb1] += U * hartree_sum(rho, orb1)
        else:
            hartree_k[orb1, orb1] += 2 * V * hartree_sum(rho, orb2)
            hartree_k[orb2, orb2] += 2 * V * hartree_sum(rho, orb1)
    return hartree_k / Nk
    
def fock_sum(rho_r, Kxmesh, delta_x):
    fock = np.sum(rho_r * np.exp(-1j * Kxmesh * delta_x)).real
    return fock

def H_fock(Kxmesh, Nk, rho, a, V):
    fock = np.zeros(rho.shape, dtype='complex')

    deltas = [0.0, a]
    for delta in deltas:
        fock[4,0] += -V * fock_sum(rho[4,0], Kxmesh, delta) * np.exp(1j * Kxmesh * delta) / Nk
        fock[4,1] += -V * fock_sum(rho[4,1], Kxmesh, delta) * np.exp(1j * Kxmesh * delta) / Nk
    fock[0,4] = fock[4,0].conjugate()
    fock[1,4] = fock[4,1].conjugate()

    deltas = [0.0, -a]
    for delta in deltas:
        fock[5,2] += -V * fock_sum(rho[5,2], Kxmesh, delta) * np.exp(1j * Kxmesh * delta) / Nk
        fock[5,3] += -V * fock_sum(rho[5,3], Kxmesh, delta) * np.exp(1j * Kxmesh * delta) / Nk
    fock[2,5] = fock[5,2].conjugate()
    fock[3,5] = fock[5,3].conjugate()
    return fock

def H_diagonalize(hop, perturb, hartree, fock, T, mu, Gamma, eps):
    Ny, Nx = fock.shape[-2:]
    H = hop + fock
    if eps != 0:
        H = H + perturb * eps
    H_full = H + hartree[:, :, np.newaxis, np.newaxis]

    energije = np.zeros((6, Ny, Nx))
    vecs     = np.zeros((6, 6, Ny, Nx), dtype=complex)
    fs       = np.zeros((6, 6, Ny, Nx))

    # eigh over last two axes — numpy handles batch automatically
    # H_full needs shape (..., M, M) so transpose to (Ny, Nx, 6, 6)
    H_batch = H_full.transpose(2, 3, 0, 1)           # (Ny, Nx, 6, 6)

    en_batch, v_batch = np.linalg.eigh(H_batch)      # (Ny, Nx, 6), (Ny, Nx, 6, 6)

    # transpose back
    energije = en_batch.transpose(2, 0, 1)            # (6, Ny, Nx)
    vecs     = v_batch.transpose(2, 3, 0, 1)          # (6, 6, Ny, Nx)

    # occupation numbers: Fermi-Dirac distribution

    fs = np.zeros((6, 6, Ny, Nx))
    if T==0:
        if Gamma==0.0:
            f_vals = np.array([1, 1, 0, 0, 0, 0], dtype=float)
        else:
            f_vals = 0.5 - np.arctan((energije-mu) / Gamma) / np.pi
    else:
        if Gamma==0.0:
            f_vals = expit(-(energije-mu) / T)
        else:
            z = 0.5 + (Gamma + 1j*(energije-mu)) / (2*np.pi*T)
            f_vals = 0.5 - np.imag(digamma(z)) / np.pi
    for b in range(6):
        fs[b, b, :, :] = f_vals[b]
    return energije, vecs, fs

''' single iteration of self-consistent loop towards solving rho = F[rho] '''
def F(rho, hop, perturb, hartree, fock, T, mu, Gamma, eps=0.0):
    _, vecs, fs = H_diagonalize(hop, perturb, hartree, fock, T, mu, Gamma, eps)
    rho_new = np.einsum('ijkl,jmkl,mnkl-> inkl', vecs, fs, np.swapaxes(vecs.conj(), 0, 1))
    err = np.max(np.abs(rho - rho_new))
    return rho_new, err

''' full self-consistent loop to find rho = F[rho]'''
def Rho_next(Kxmesh, rho, hop, perturb, hartree, fock, a, U, V, T, mu, Gamma, maxiter, mix, epsilon, eps0=0.0, N_epsilon=5, hartree_list=None):
    Ny, Nx = Kxmesh.shape
    Nk = Ny * Nx
    err, N_iters = 1.0, 0
    while (err > epsilon or (eps0 != 0 and N_iters <= N_epsilon)) and N_iters < maxiter:
        eps = eps0 if N_iters < N_epsilon else 0
        rho_new, err = F(rho, hop, perturb, hartree, fock, T, mu, Gamma, eps=eps)
        rho = rho_new * mix + rho * (1 - mix)
        fock = H_fock(Kxmesh, Nk, rho, a, V)
        hartree = H_hartree(rho, Nk, U, V, hartree_list)
        N_iters += 1
    #rho_dag = np.swapaxes(rho.conj(), 0, 1)
    #relative_error = np.linalg.norm(rho - rho_dag) / max(np.linalg.norm(rho), 1e-30)
    #if relative_error > 1e-15:
    #    print("rho Hermiticity error:", relative_error, flush=True)
    rho = 0.5 * (rho + np.swapaxes(rho.conj(), 0, 1))
    fock = H_fock(Kxmesh, Nk, rho, a, V)
    hartree = H_hartree(rho, Nk, U, V, hartree_list)
    energije, vecs, fs = H_diagonalize(hop, perturb, hartree, fock, T, mu, Gamma, eps=0)
    # Report the residual of the returned, unperturbed state, after mixing.
    rho_check = np.einsum('iakl,abkl,jbkl->ijkl', vecs, fs, vecs.conj())
    err = np.max(np.abs(rho - rho_check))
    n = Occupation(rho)
    return rho, energije, fs, vecs, fock, hartree, err, n

def Occupation(rho):
    return (np.sum(np.diag(np.einsum('ijkl->ij', rho)))/(np.prod(rho.shape[-2:]))).real
    
def GS(Kxmesh, rho, hop, perturb, hartree, fock, mu, Gamma, eps0, a, U, V, epsilon=1e-10, maxiter=1000, N_epsilon=5, T=0, hartree_list=None):
    rho, energije, fs, vecs, fock, hartree, err, n = Rho_next(Kxmesh, rho, hop, perturb, hartree, fock, a, U, V, T, mu, Gamma, maxiter=maxiter, mix=1.0, epsilon=epsilon, eps0=eps0, N_epsilon=N_epsilon, hartree_list=hartree_list)
    return rho, energije, fs, vecs, fock, hartree, err, n

def NewMu(n_target, Kxmesh, rho, hop, perturb, hartree, fock, a, U, V, T, mu, dmu, Gamma, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials, faktor1=0.001, hartree_list=None, *, eps0=0.0, N_epsilon=5, max_step=None):
    """Find fixed filling with a damped slope estimate, expansion and bisection.

    Each bracket endpoint is an evaluated, self-consistent state. Nearby trials
    reuse rho and both mean fields together. maxiter_last is the continuation
    budget for each of at most two continuations if a short solve has not
    converged. The second continuation reduces mixing for stability; no
    unconverged density is used.
    max_trials bounds each of bracket expansion and bisection. max_step caps
    the initial slope estimate (default: the current spectral span).
    """
    if not (np.isfinite(T) and T >= 0 and np.isfinite(Gamma) and Gamma >= 0):
        raise ValueError("T and Gamma must be finite and nonnegative")
    if not (np.isfinite(n_target) and 0 < n_target < 6):
        raise ValueError("n_target must lie strictly between 0 and 6")
    if T == 0 and Gamma == 0:
        raise ValueError("Use GS for T=Gamma=0; its occupations fix the filling at two")
    if not (np.isfinite(mu) and np.isfinite(dmu) and dmu != 0):
        raise ValueError("mu must be finite and dmu must be finite and nonzero")
    if not (0 < mix <= 1 and np.isfinite(mix2) and mix2 > 0
            and np.isfinite(mix3) and mix3 > 0):
        raise ValueError("Require 0 < mix <= 1 and positive finite mix2, mix3")
    if not (np.isfinite(eps_last) and eps_last > 0 and np.isfinite(n_pass) and n_pass > 0):
        raise ValueError("Convergence tolerances must be finite and positive")
    if any(not isinstance(x, (int, np.integer)) or x < 1
           for x in (maxiter, maxiter_last, max_trials)):
        raise ValueError("Iteration budgets must be positive integers")
    if not (np.isfinite(faktor1) and faktor1 > 0):
        raise ValueError("faktor1 must be finite and positive")
    if max_step is not None and not (np.isfinite(max_step) and max_step > 0):
        raise ValueError("max_step must be finite and positive")

    # A point is (mu, complete Rho_next result). Retain only bracket endpoints
    # and the latest point, rather than caching large arrays for every trial.
    lower = upper = latest = None
    current_mix = mix

    def evaluate(trial_mu):
        nonlocal latest, lower, upper, current_mix
        if not np.isfinite(trial_mu):
            raise RuntimeError("Chemical-potential search exceeded finite bounds")
        seeds = [p for p in (lower, upper, latest) if p is not None]
        if seeds:
            seed = min(seeds, key=lambda p: abs(p[0] - trial_mu))[1]
            rho_seed, fock_seed, hartree_seed = seed[0], seed[4], seed[5]
        else:
            rho_seed, fock_seed, hartree_seed = rho, fock, hartree
        state = Rho_next(Kxmesh, rho_seed, hop, perturb, hartree_seed, fock_seed,
                         a, U, V, T, trial_mu, Gamma, maxiter, current_mix, eps_last,
                         eps0=eps0 if not seeds else 0.0,
                         N_epsilon=N_epsilon, hartree_list=hartree_list)
        for retry in range(2):
            if not np.isfinite(state[-2]) or state[-2] <= eps_last:
                break
            if retry == 1:
                current_mix = max(mix / 4, current_mix / 2)
            state = Rho_next(Kxmesh, state[0], hop, perturb, state[5], state[4],
                             a, U, V, T, trial_mu, Gamma, maxiter_last, current_mix,
                             eps_last, hartree_list=hartree_list)
        err, n = state[-2:]
        if not (np.isfinite(err) and np.isfinite(n) and err <= eps_last):
            raise RuntimeError(f"Density did not converge at mu={trial_mu:.12g}: "
                               f"err={err:.3g}, tolerance={eps_last:.3g}")
        latest = (trial_mu, state)
        if n < n_target and (lower is None or trial_mu > lower[0]):
            lower = latest
        elif n > n_target and (upper is None or trial_mu < upper[0]):
            upper = latest
        if lower is not None and upper is not None and lower[0] >= upper[0]:
            raise RuntimeError("Inconsistent filling bracket; check convergence or a change of mean-field branch")
        return latest

    def accepted(point):
        return abs(point[1][-1] - n_target) <= n_pass

    def result(point):
        return (*point[1], point[0])

    first = evaluate(mu)
    if accepted(first):
        return result(first)
    # Probe toward the target; this often brackets the root immediately.
    direction = np.sign(n_target - first[1][-1])
    probe = evaluate(mu + direction * abs(dmu))
    if accepted(probe):
        return result(probe)
    dn = probe[1][-1] - first[1][-1]
    chi = dn / (probe[0] - first[0])
    step_limit = max(abs(dmu), faktor1, float(np.ptp(first[1][1]))) if max_step is None else max_step
    step = max(abs(dmu), faktor1)
    # The trace sums six diagonal entries, so reject differences comparable
    # to the self-consistency error rather than amplifying numerical noise.
    if np.isfinite(chi) and chi > 0 and abs(dn) > 12 * eps_last:
        correction = (n_target - first[1][-1]) / chi
        step = max(step, min(abs(correction) * mix3, step_limit))
        trial_mu = first[0] + np.clip(mix2 * correction, -step_limit, step_limit)
        inside = lower is None or upper is None or lower[0] < trial_mu < upper[0]
        if inside and trial_mu not in (first[0], probe[0]):
            point = evaluate(trial_mu)
            if accepted(point):
                return result(point)
    # Release the initial spectra once their slope has been used.
    del first, probe

    for _ in range(max_trials):
        if lower is not None and upper is not None:
            break
        anchor = lower if lower is not None else upper
        direction = 1.0 if lower is not None else -1.0
        point = evaluate(anchor[0] + direction * step)
        if accepted(point):
            return result(point)
        step *= 2.0
    if lower is None or upper is None:
        raise RuntimeError(f"Could not bracket chemical potential after {max_trials} expansion trials")

    for _ in range(max_trials):
        trial_mu = lower[0] + 0.5 * (upper[0] - lower[0])
        if trial_mu in (lower[0], upper[0]):
            break
        point = evaluate(trial_mu)
        if accepted(point):
            return result(point)
    raise RuntimeError(f"Chemical-potential search did not reach filling tolerance {n_pass:.3g} "
                       f"after {max_trials} bisections; last n={latest[1][-1]:.12g}")


def broadened_mu_guess(energies, Gamma, n_target=2.0, *, mu0=None, n_pass=1e-12):
    """Solve the T=0 filling equation for a fixed spectrum.

    energies has shape (bands, ...momentum grid...). Safeguarded Newton steps
    evaluate only occupations and their derivative, with bisection as fallback.
    mu0 is used when supplied; no diagonalization or mean-field solve occurs.
    For a self-consistent calculation this solves only the current spectrum.
    """
    energies = np.asarray(energies, dtype=float)
    if energies.ndim < 1 or energies.size == 0 or not np.all(np.isfinite(energies)):
        raise ValueError("energies must be a finite, nonempty band spectrum")
    if not np.isfinite(Gamma) or Gamma <= 0:
        raise ValueError("The broadened initial guess requires Gamma > 0")
    if not np.isfinite(n_target) or not 0 < n_target < energies.shape[0]:
        raise ValueError("n_target must lie between zero and the number of bands")
    if mu0 is not None and not np.isfinite(mu0):
        raise ValueError("mu0 must be finite")
    if not np.isfinite(n_pass) or n_pass <= 0:
        raise ValueError("n_pass must be finite and positive")
    nk = energies.size / energies.shape[0]

    def filling(mu):
        return np.sum(np.arctan2(Gamma, energies - mu)) / (np.pi * nk)

    lo, hi = float(np.min(energies)), float(np.max(energies))
    step = max(hi - lo, Gamma)
    for _ in range(100):
        if filling(lo) <= n_target <= filling(hi):
            break
        lo -= step
        hi += step
        step *= 2
    else:
        raise RuntimeError("Could not bracket the fixed-spectrum filling")
    mu = float(mu0) if mu0 is not None and lo <= mu0 <= hi else lo + 0.5 * (hi - lo)
    for iteration in range(200):
        n = filling(mu)
        if abs(n - n_target) <= n_pass:
            return mu
        if n < n_target:
            lo = mu
        else:
            hi = mu
        midpoint = lo + 0.5 * (hi - lo)
        if midpoint == lo or midpoint == hi:
            raise RuntimeError("Fixed-spectrum filling tolerance is below numerical resolution")
        delta = energies - mu
        slope = np.sum(Gamma / (delta*delta + Gamma*Gamma)) / (np.pi * nk)
        candidate = mu - (n - n_target) / slope if np.isfinite(slope) and slope > 0 else np.nan
        # Force occasional bisection so even extreme starting guesses make
        # bracket progress; near the solution Newton normally needs 1–3 steps.
        mu = candidate if (iteration % 4 != 3 and np.isfinite(candidate)
                           and lo < candidate < hi and candidate != mu) else midpoint
    raise RuntimeError("Fixed-spectrum chemical-potential guess did not converge")


''' chemical potential in ground state T=0 '''
def ground_state_fixed_filling(Kxmesh, rho, hop, perturb, hartree, fock, a, U, V, T, mu0, dmu, Gamma, maxiter, mix, epsilon, eps0, N_epsilon, hartree_list,
                               n_target=2.0, n_pass=1e-6, *, method="fixed_filling"):
    """At T=0 solve mu on each current spectrum inside the density iteration.

    This keeps the density close to the target without a full mean-field solve
    for every trial mu. mu0 seeds the first scalar solve; later solves reuse mu.
    method='outer' retains NewMu for comparison. Finite T still uses NewMu.
    """
    if method not in ("fixed_filling", "outer"):
        raise ValueError("method must be 'fixed_filling' or 'outer'")
    if T == 0 and Gamma == 0:
        if n_target != 2.0:
            raise ValueError("The sharp ground-state occupations support only n_target=2")
        state = GS(Kxmesh, rho, hop, perturb, hartree, fock, mu0, Gamma,
                   eps0, a, U, V, epsilon=epsilon, maxiter=maxiter,
                   N_epsilon=N_epsilon, hartree_list=hartree_list)
        if not np.isfinite(state[-2]) or state[-2] > epsilon:
            raise RuntimeError("Sharp ground-state density did not converge")
        energies = state[1]
        valence, conduction = np.max(energies[1]), np.min(energies[2])
        if valence > conduction:
            raise ValueError("Two fully occupied bands do not define a gapped ground state")
        return (*state, 0.5 * (valence + conduction))
    if T == 0 and Gamma > 0 and method == "fixed_filling":
        if not (0 < mix <= 1 and np.isfinite(epsilon) and epsilon > 0
                and np.isfinite(n_pass) and n_pass > 0):
            raise ValueError("Require 0 < mix <= 1 and positive finite convergence tolerances")
        if not isinstance(maxiter, (int, np.integer)) or maxiter < 1:
            raise ValueError("maxiter must be a positive integer")
        mu = mu0
        rho = np.array(rho, dtype=complex, copy=True)
        rho = .5 * (rho + np.swapaxes(rho.conj(), 0, 1))
        nk = Kxmesh.size
        # Derive both fields from the same density, including after a restart.
        fock = H_fock(Kxmesh, nk, rho, a, V)
        hartree = H_hartree(rho, nk, U, V, hartree_list)
        number_tol = min(n_pass * .1, epsilon * .1)
        idx = np.arange(rho.shape[0])
        for attempt in range(3):
            current_mix = mix / (2**attempt)
            for iteration in range(maxiter):
                eps = eps0 if attempt == 0 and iteration < N_epsilon else 0.
                energije, vecs, fs = H_diagonalize(hop, perturb, hartree, fock,
                                                  0., 0. if mu is None else mu, Gamma, eps)
                mu = broadened_mu_guess(energije, Gamma, n_target,
                                        mu0=mu, n_pass=number_tol)
                occupations = np.arctan2(Gamma, energije-mu) / np.pi
                fs[idx, idx] = occupations
                rho_new = np.einsum('iakl,akl,jakl->ijkl', vecs, occupations, vecs.conj())
                err = np.max(np.abs(rho_new-rho))
                n = Occupation(rho)
                if not np.isfinite(err) or not np.isfinite(n):
                    raise RuntimeError("Nonfinite density in fixed-filling ground-state solve")
                if eps == 0 and err <= epsilon and abs(n-n_target) <= n_pass:
                    return rho, energije, fs, vecs, fock, hartree, err, n, mu
                rho = (1-current_mix)*rho + current_mix*rho_new
                rho = .5 * (rho + np.swapaxes(rho.conj(), 0, 1))
                fock = H_fock(Kxmesh, nk, rho, a, V)
                hartree = H_hartree(rho, nk, U, V, hartree_list)
        raise RuntimeError(f"Fixed-filling ground state did not converge: err={err:.3g}, "
                           f"|n-n_target|={abs(n-n_target):.3g}")
    if T == 0 and Gamma > 0 and mu0 is None:
        energies, _, _ = H_diagonalize(hop, perturb, hartree, fock,
                                       T, 0., Gamma, eps=0.0)
        mu0 = broadened_mu_guess(energies, Gamma, n_target)
    return NewMu(n_target, Kxmesh, rho, hop, perturb, hartree, fock, a, U, V,
                 T, mu0, dmu, Gamma, maxiter, maxiter, epsilon, mix,
                 1.0, 1.5, n_pass, 100, hartree_list=hartree_list,
                 eps0=eps0, N_epsilon=N_epsilon)

''' density of states '''
@njit(parallel=False, cache=True)
def DoS(Kymesh, Kxmesh, energije, omegas, mu, velocity_x, velocity_y, faktor=1.):
    Ny, Nx = Kymesh.shape
    Nk = Ny*Nx
    dKy, dKx = Kymesh[:,0][1] - Kymesh[:,0][0], Kxmesh[0][1] - Kxmesh[0][0]

    domega = omegas[1] - omegas[0]
    dos = np.zeros((6, omegas.shape[0]))
    v_max = np.array([np.max(np.abs(velocity_x)), np.max(np.abs(velocity_y))])
    sigma = np.max(np.array([np.sqrt(v_max[0] * domega * dKx) * faktor, np.sqrt(v_max[1] * domega * dKy) * faktor]))

    for m in [0, Ny//2]:
        for n in range(Nx):
            for orb in range(6):
                dos[orb] += 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-(omegas - (energije[orb,m,n] - mu))**2/(2*sigma**2))
    for n in [0, Nx//2]:
        for m in range(Ny):
            for orb in range(6):
                dos[orb] += 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-(omegas - (energije[orb,m,n] - mu))**2/(2*sigma**2))
    for m in range(Ny):
        for n in prange(1,Nx//2):
            if m not in [0, Ny//2]:
                for orb in range(6):
                    dos[orb] += 2. * 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-(omegas - (energije[orb,m,n] - mu))**2/(2*sigma**2))
    return dos * 2 / Nk # factor 2 for spin

def colors(vecs):
    return np.einsum('ijkl->jkl', np.abs(vecs[:4,:,:,:])**2)
