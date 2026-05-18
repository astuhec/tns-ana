import numpy as np
from scipy.special import expit
from numba import njit, prange
import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
import os

os.chdir("/Users/ana/Desktop/tns-ana/main-programs/")

def Rho0(Ny, Nx):
    rho0 = np.zeros((6, 6, Ny, Nx), dtype='complex')
    
    rho0[5, 5, :, :] = 1.0
    rho0[4, 4, :, :] = 1.0

    return rho0

def Rhoinfty(Ny, Nx):
    nx = Nx//2 + 1
    rho0 = np.zeros((6, 6, Ny, nx), dtype='complex')
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

def H_hopping(Kymesh, Kxmesh, a, b,file = 'parametri-kinetic.txt', faktor=1.):
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

def H_perturb(Kymesh, kxmesh, a, b, file = 'perturbacija.txt'):
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

file = 'parametri-interaction.txt'
hartree_list = []
with open(file) as f:
    for line in f:
        [orb1, orb2] = list(map(float, line.split()))[-2:]
        orb1, orb2 = int(orb1-1), int(orb2-1)
        hartree_list.append([orb1,orb2])

def H_hartree(rho, Nk, U, V,):
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

    deltas = [0., a]
    for delta in deltas:
        fock[4,0] += -V * fock_sum(rho[4,0], Kxmesh, delta) * np.exp(1j * Kxmesh * delta) / Nk
        fock[4,1] += -V * fock_sum(rho[4,1], Kxmesh, delta) * np.exp(1j * Kxmesh * delta) / Nk
    fock[0,4] = fock[4,0].conjugate()
    fock[1,4] = fock[4,1].conjugate()

    deltas = [0., -a]
    for delta in deltas:
        fock[5,2] += -V * fock_sum(rho[5,2], Kxmesh, delta) * np.exp(1j * Kxmesh * delta) / Nk
        fock[5,3] += -V * fock_sum(rho[5,3], Kxmesh, delta) * np.exp(1j * Kxmesh * delta) / Nk
    fock[2,5] = fock[5,2].conjugate()
    fock[3,5] = fock[5,3].conjugate()
    return fock

def H_diagonalize2(hop, perturb, hartree, fock, T, mu, eps):
    Ny, Nx = fock.shape[-2:]
    H = hop + fock
    if eps != 0:
        H = H + perturb * eps

    energije = np.zeros((6, Ny, Nx))
    vecs     = np.zeros((6, 6, Ny, Nx), dtype=complex)
    fs       = np.zeros((6, 6, Ny, Nx))

    # add hartree once — broadcast over k-points
    H_full = H + hartree[:, :, np.newaxis, np.newaxis]  # shape (6,6,Ny,Nx)

    # eigh over last two axes — numpy handles batch automatically
    # H_full needs shape (..., M, M) so transpose to (Ny, Nx, 6, 6)
    H_batch = H_full.transpose(2, 3, 0, 1)           # (Ny, Nx, 6, 6)

    en_batch, v_batch = np.linalg.eigh(H_batch)      # (Ny, Nx, 6), (Ny, Nx, 6, 6)

    # transpose back
    energije = en_batch.transpose(2, 0, 1)            # (6, Ny, Nx)
    vecs     = v_batch.transpose(2, 3, 0, 1)          # (6, 6, Ny, Nx)

    # Fermi factors
    if T == 0:
        f_vals = np.array([1, 1, 0, 0, 0, 0], dtype=float)
        fs = np.zeros((6, 6, Ny, Nx))
        for b in range(6):
            fs[b, b, :, :] = f_vals[b]
    elif T == 'infty':
        f_vals = np.array([1, 1, 1, 1, 1, 1], dtype=float) / 3
        fs = np.zeros((6, 6, Ny, Nx))
        for b in range(6):
            fs[b, b, :, :] = f_vals[b]
    else:
        # energije shape (6, Ny, Nx) → expit gives same shape
        f_en = expit(-(energije - mu) / T)            # (6, Ny, Nx)
        # fill diagonal: fs[b, b, m, n] = f_en[b, m, n]
        idx = np.arange(6)
        fs[idx, idx, :, :] = f_en                     # vectorized diagonal fill
        
    return energije, vecs, fs


def F(rho, hop, perturb, hartree, fock, T, mu, eps=0, colors=False, occupation=False, vectors=False):
    energije, vecs, fs = H_diagonalize2(hop, perturb, hartree, fock, T, mu, eps)
    rho_new = np.einsum('ijkl,jmkl,mnkl-> inkl', vecs, fs, np.swapaxes(vecs.conj(), 0, 1))
    if colors :
        barve = np.einsum('ijkl->jkl', np.abs(vecs[:4, :, :, :])**2)
        return rho_new, energije, np.max(np.abs(rho - rho_new)), barve
    if not occupation: return rho_new, energije, np.max(np.abs(rho - rho_new))
    elif occupation and vectors:
        return rho_new, energije, fs, vecs, np.max(np.abs(rho - rho_new))
    
def Rho_next(Kxmesh, rho, hop, perturb, hartree, fock, a, U, V, T, mu, maxiter, mix, epsilon, eps0=0.0, N_epsilon=5):
    Ny, Nx = Kxmesh.shape
    Nk = Ny * Nx
    err, N_iters = 1, 0
    while err > epsilon and N_iters < maxiter:
        if N_iters < N_epsilon: eps = eps0
        else: eps0 = 0
        rho_new, _, err = F(rho, hop, perturb, hartree, fock, T, mu, eps=eps)
        rho = rho_new * mix + rho * (1 - mix)
        fock = H_fock(Kxmesh, Nk, rho, a, V)
        hartree = H_hartree(rho, Nk, U, V)
        rho = rho_new * mix + rho * (1 - mix)
        N_iters += 1
    rho, energije, fs, vecs, err = F(rho, hop, perturb, hartree, fock, T, mu, occupation=True, vectors=True)
    return rho, energije, fs, vecs, fock, hartree, err, Occupation(rho)

def Occupation(rho):
    return (np.sum(np.diag(np.einsum('ijkl->ij', rho)))/(np.prod(rho.shape[-2:]))).real
    
def GS(Kxmesh, rho, hop, perturb, hartree, fock, mu, eps0, a, U, V, epsilon=1e-10, maxiter=1000, N_epsilon=5):
    Ny, Nx = Kxmesh.shape
    Nk = Ny*Nx
    err, N_iters = 1, 0
    while err > epsilon and N_iters < maxiter:
        if N_iters < N_epsilon: eps = eps0
        else: eps = 0
        rho, _, err = F(rho, hop, perturb, hartree, fock, 0, mu, eps=eps)
        fock = H_fock(Kxmesh, Nk, rho, a, V)
        hartree = H_hartree(rho, Nk, U, V)
        N_iters += 1
    rho, energije, fs, vecs, err = F(rho, hop, perturb, hartree, fock, 0, mu, eps=eps, occupation=True, vectors=True)
    #print(f'Found ground state with error={err}, occupation error={np.abs(Occupation(rho)-2)}')
    return rho, energije, fs, vecs, err, Occupation(rho), fock, hartree

''''''''''''''''''''''''''

def NewOccupation(Kxmesh, rho, hop, perturb, hartree, fock, a, U, V, T, mu, n_target, Dmu, steps):
    Ny, Nx = Kxmesh.shape
    Nk = Ny*Nx
    Dmu = Dmu*np.sign(n_target - 2)
    dmu = Dmu / steps
    for i in range(steps):
        mu = mu + dmu
        rho_new, _, _ = F(rho, hop, perturb, hartree, fock, T, mu + i*dmu, eps=0.)
        rho = 0.5 * (rho_new + rho)
        fock = H_fock(Kxmesh, Nk, rho, a, V)
        hartree = H_hartree(rho, Nk, U, V)
        n = Occupation(rho)
        print(n)
        if np.sign(n_target - 2) * np.sign(n - n_target) == +1: break
    return rho, hartree, fock, n, mu

def NewMu(n_target, Kxmesh, rho, hop, perturb, hartree, fock, a, U, V, T, mu, dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials, faktor1=0.001):
    _, _, _, _, _, _, err_a, n_a = Rho_next(Kxmesh, rho, hop, perturb, hartree, fock, a, U, V, T, mu, maxiter, mix, eps_last)
    #if np.abs(n_a - n_target) < n_pass and err_a < eps_last:
    #    return rho_a, energije_a, fs_a, vecs_a, fock_a, hartree_a, err_a, n_a, mu
    n_b = Rho_next(Kxmesh, rho, hop, perturb, hartree, fock, a, U, V, T, mu + dmu, maxiter, mix, eps_last)[-1]
    chi = (n_b - n_a)/dmu
    
    # Improved handling for very weak chi (narrow search space or plateau in n(mu))
    # This is common when n is already very close to n_target
    if abs(chi) < 1e-5:
        # Very weak response: use smaller step to avoid overshooting
        step_direction = np.sign(n_a - n_target)
        mu = mu - 0.1 * dmu * step_direction
    elif chi != 0:
        mu = mu - mix2 * (n_a - n_target)/np.abs(chi)

    pogoj = False
    koraki = 0
    if np.abs(chi) > 0: faktor = (n_a - n_target)/chi * mix3
    else: faktor = faktor1
    if chi >= 0:
        if n_a >= n_target:
            sign = -1
        elif n_a < n_target: sign = +1
    elif chi < 0:
        if n_a >= n_target: sign = +1
        elif n_a < n_target: sign = -1
        
    sgns = np.ones(2) * np.sign(n_a - n_target)
    ns = np.array([0, n_a])
    mus = [0, mu]
    enough = False
    while sgns[0] == sgns[1]:
        if np.abs(n_a - n_target) < n_pass and err_a < eps_last:
            enough = True
            break
        _, _, _, _, _, _, err_b, n_b = Rho_next(Kxmesh, rho, hop, perturb, hartree, fock, a, U, V, T, mu + faktor*koraki*sign, maxiter, mix, eps_last)
        #if np.abs(n_b - n_target) < n_pass and err_b < eps_last: return rho_b, energije_b, fs_b, vecs_b, fock_b, hartree_b, err_b, n_b,  mu + faktor*koraki*sign
        ns[0] = n_b
        mus[0] = mu + faktor*koraki*sign
        sgns[1] = np.sign(n_b - n_target)
        if sgns[0] != sgns[1]: break
        if n_b < n_target and n_b < ns[1]: sign *= -1
        if n_b > n_target and n_b > ns[1]: sign *= -1
        ns = np.roll(ns, 1)
        mus = np.roll(mus, 1)
        sgns[1] = np.sign(n_b - n_target)
        koraki +=1
        if np.abs(n_b - n_target) < n_pass and err_b < eps_last:
            enough = True
            mu_mid = mu + faktor*koraki*sign
            break
        
    mus = np.sort(np.array([mu + faktor*koraki*sign, mu + faktor*(koraki-1)*sign]))
    ns = np.sort(np.array(ns))

    trials = 0
    while pogoj == False:
        mu_mid = (mus[0] + mus[1])/2
        if enough == True:
            break   
        n_mid = Rho_next(Kxmesh, rho, hop, perturb, hartree, fock, a, U, V, T, mu_mid, maxiter, mix, eps_last)[-1]
        if n_mid > n_target: mus[1] = mu_mid
        elif n_mid < n_target: mus[0] = mu_mid
        if np.abs(n_mid - n_target) < n_pass: break
        trials += 1 
        if trials > max_trials: break
    rho, energije, fs, vecs, fock, hartree, err, n = Rho_next(Kxmesh, rho, hop, perturb, hartree, fock, a, U, V, T, mu_mid, maxiter_last, mix, eps_last)
    return rho, energije, fs, vecs, fock, hartree, err, n, mu_mid

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