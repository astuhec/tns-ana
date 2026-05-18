import numpy as np
import os, sys
from numba import njit, prange
import warnings
import scipy.linalg as LA
from numba.core.errors import NumbaPerformanceWarning
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
#os.chdir("/Users/ana/Desktop/tns-ana/main-programs/")

import tns_helpers as helpers

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

def kinetic(file="parametri-kinetic.txt", extend=None, faktor=1.0):
    seznam = []
    with open(file, 'r') as f:
        for line in f:
            [x, y, orb1, orb2, t] = list(map(float, line.split()))
            if orb1 != orb2: t = faktor * t
            seznam.append([x, y, orb1, orb2, t])
            if extend == True:
                if orb1 != orb2:
                    seznam.append([-x, -y, orb2, orb1, t])
    return np.array(seznam)

def interaction(file = "parametri-interaction.txt"):
    seznam = []
    with open(file, 'r') as f:
        for line in f:
            [x, y, orb1, orb2] = list(map(float, line.split()))
            seznam.append([x, y, orb1, orb2])
            if orb1 != orb2:
                seznam.append([-x, -y, orb2, orb1])
    return np.array(seznam)

def positions(a, b, b2):
    return np.array([[0.,0.],
                     [-a/4, b/2 - b2],
                    [-a/4,b2],
                    [a/4,-b2],
                    [a/4, -b/2 + b2],
                    [a/4,b/4],
                    [-a/4, -b/4]])
#xcor_[0]=1.752/a_;xcor_[1]=1.752/a_;xcor_[2]=3.5035/a_;xcor_[3]=3.5035/a_;xcor_[4]=3.5035/a_;xcor_[5]=1.752/a_
#ycor_[0]=13.86/c_;ycor_[1]=9.807/c_;ycor_[2]=5.982/c_;ycor_[3]=1.927/c_;ycor_[4]=11.92/c_;ycor_[5]=3.94/c_

''' matrix for number density operator '''
def j_tok(Kymesh, Kxmesh, a, b, b2, file):
    pos = positions(a, b, b2)
    Ny, Nx = Kymesh.shape
    jx = np.zeros((6, 6, Ny, Nx), dtype=np.complex128)
    jy = np.copy(jx)

    for line in file:
        x, y, orb1, orb2, t = line
        x, y, orb1, orb2, t = float(x), float(y), int(orb1), int(orb2), float(t)
        if orb1 == orb2 and (x,y) == (0,0): pass # this is onsite energy, does not contribute to j
        else:
            osnova = 1j * t * np.exp(-1j * (Kxmesh * x * a + Kymesh * y * b))
            lega = pos[orb2] - pos[orb1] - np.array([x*a, y*b])
            ad_x = osnova * lega[0]
            ad_y = osnova * lega[1]

            jx[orb1 - 1, orb2 - 1] += ad_x
            if orb1 != orb2:
                jx[orb2 - 1, orb1 - 1] += ad_x.conjugate() 
            jy[orb1 - 1, orb2 - 1] += ad_y
            if orb1 != orb2:
                jy[orb2 - 1, orb1 - 1] += ad_y.conjugate()
    jmatrix = np.zeros((2,6,6,Ny,Nx), dtype=np.complex128)
    jmatrix[0] = jx
    jmatrix[1] = jy
    return jmatrix

''' def j_Fock(rho, Kymesh, Kxmesh, a, b, b2, V):
    pos = positions(a, b, b2)
    Ny, Nx = Kymesh.shape
    Nk = Ny * Nx
    tok = np.zeros((2, 6, 6, Ny, Nx), dtype=np.complex128)

    deltas = [np.array([0., 0.]), np.array([-a, 0.])]
    for delta in deltas:
        for orb in [1,2]:
            lega = pos[orb] - pos[5] - delta
            t = -V * np.sum(rho[orb-1,4] * np.exp(-1j*(Kxmesh*delta[0] + Kymesh*delta[1]))) / Nk
            osnova = 1j * t * np.exp(-1j*(Kxmesh*delta[0] + Kymesh*delta[1]))
            for nu in range(2):
                tok[nu,4,orb-1] += osnova * lega[nu]
                tok[nu,orb-1,4] += (osnova * lega[nu]).conjugate()
    deltas = [np.array([0., 0.]), np.array([a, 0.])]
    for delta in deltas:
        for orb in [3,4]:
            lega = pos[orb] - pos[6] - delta
            t = -V * np.sum(rho[orb-1,5] * np.exp(-1j*(Kxmesh*delta[0] + Kymesh*delta[1]))) / Nk
            osnova = 1j * t * np.exp(-1j*(Kxmesh*delta[0] + Kymesh*delta[1]))
            for nu in range(2):
                tok[nu,5,orb-1] += osnova * lega[nu]
                tok[nu,orb-1,5] += (osnova * lega[nu]).conjugate()
    return tok '''

# HF stands for Hellmann-Feynmann
def velocity_HF(Kymesh, Kxmesh, a, b, file):
    Ny, Nx = Kymesh.shape
    dH_dk = np.zeros((2, 6, 6, Ny, Nx), dtype='complex')
    for line in file:
        [x, y, orb1, orb2, t] = line
        orb1, orb2 = int(orb1), int(orb2)
        ad = -1j * t * np.exp(-1j*(Kxmesh * x * a + Kymesh * y * b))
        position_x = x * a
        position_y = y * b
        dH_dk[0, orb1 - 1, orb2 - 1] += position_x * ad
        dH_dk[1, orb1 - 1, orb2 - 1] += position_y * ad
        if orb1 != orb2:
            dH_dk[0, orb2 - 1, orb1 - 1] += position_x * ad.conjugate()
            dH_dk[1, orb2 - 1, orb1 - 1] += position_y * ad.conjugate()
    return dH_dk

def fock_sum(rho_r, Kxmesh, delta_x):
    fock = np.sum(rho_r * np.exp(-1j * Kxmesh * delta_x)).real
    return fock

def velocity_fock_HF(rho, Kymesh, Kxmesh, a, V):
    Ny, Nx = Kymesh.shape
    Nk = Ny*Nx

    fock_velocity = np.zeros(rho.shape, dtype='complex')

    deltas = [0., a]
    for delta in deltas:
        fock_velocity[4,0] += -V * fock_sum(rho[4,0], Kxmesh, delta) * np.exp(1j * Kxmesh * delta) / Nk * 1j * delta
        fock_velocity[4,1] += -V * fock_sum(rho[4,1], Kxmesh, delta) * np.exp(1j * Kxmesh * delta) / Nk * 1j * delta
    fock_velocity[0,4] = fock_velocity[4,0].conjugate()
    fock_velocity[1,4] = fock_velocity[4,1].conjugate()

    deltas = [0., -a]
    for delta in deltas:
        fock_velocity[5,2] += -V * fock_sum(rho[5,2], Kxmesh, delta) * np.exp(1j * Kxmesh * delta) / Nk * 1j * delta
        fock_velocity[5,3] += -V * fock_sum(rho[5,3], Kxmesh, delta) * np.exp(1j * Kxmesh * delta) / Nk * 1j * delta
    fock_velocity[2,5] = fock_velocity[5,2].conjugate()
    fock_velocity[3,5] = fock_velocity[5,3].conjugate()

    return fock_velocity

@njit
def expit_stable(x):
    if x >= 0:
        # large positive x: exp(-x) is small, safe
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        # large negative x: exp(x) is small, safe
        z = np.exp(x)
        return z / (1.0 + z)
@njit
def expit_prime_stable(x):
    s = 1/ (1 + np.exp(-x))
    return s * (1.0 - s)

@njit
def fd_1(omega, T): return -1/(4*T)/np.cosh(omega/(2*T))**2

''' approximation for Dirac delta function '''
def delta_approximation(x, width, shape='Gaussian'):
    if shape == 'Gaussian':
        return 1/(2*np.pi*width**2)**0.5 * np.exp(-x**2/(2*width**2))
    elif shape == 'Lorentzian':
        return 1/np.pi * width / (x**2 + width**2)
    
@njit(parallel=False, cache=True)
def phi_boltzmann(Kymesh, Kxmesh, velocity_x, velocity_y, energije, omegas, mu, faktor=1., shape='Gaussian'):
    Ny, Nx = Kymesh.shape
    Nk = Ny*Nx
    dKy = Kymesh[:,0][1] - Kymesh[:,0][0]
    dKx = Kxmesh[0][1] - Kxmesh[0][0]

    domega = omegas[1] - omegas[0]
    transportna_x = np.zeros(omegas.shape[0])
    transportna_y = np.zeros(omegas.shape[0])

    v_max_x = np.max(np.abs(velocity_x))
    v_max_y = np.max(np.abs(velocity_y))

    sigma_x = np.sqrt(v_max_x * domega * dKx) * faktor
    sigma_y = np.sqrt(v_max_y * domega * dKy) * faktor

    for m in range(Ny):
        for n in [0,Nx//2]:
            for orb in prange(6):
                transportna_x += delta_approximation(omegas - energije[orb,m,n] + mu, sigma_x, shape) * velocity_x[orb,m,n]**2
                transportna_y += delta_approximation(omegas - energije[orb,m,n] + mu, sigma_y, shape) * velocity_y[orb,m,n]**2

    for m in prange(Ny):
        for n in range(1,Nx//2):
                for orb in range(6):
                    transportna_x += 2 * delta_approximation(omegas - energije[orb,m,n] + mu, sigma_x, shape) * velocity_x[orb,m,n]**2
                    transportna_y += 2 * delta_approximation(omegas - energije[orb,m,n] + mu, sigma_y, shape) * velocity_y[orb,m,n]**2
    
    return transportna_x * 2 / Nk, transportna_y * 2 / Nk # factor 2 for spin

def kahan_sum(vals):
    total = 0.0
    c = 0.0
    for x in vals:
        y = x - c
        t = total + y
        c = (t - total) - y
        total = t
    return total

def Kn_boltzmann(velocity_x, velocity_y, energije, mu, T):
    Norb, Ny, Nx = energije.shape
    Nk = Ny*Nx
    
    fd1 = -np.zeros((Norb, Ny, Nx//2+1))
    for orb in range(Norb):
        fd1[orb,:,:] = -fd_1(energije[orb,:,:Nx//2+1] - mu, T)

    K0_x = []
    K0_y = []
    K1_x = []
    K1_y = []

    for orb in range(6):
        velx = velocity_x[orb]
        vely = velocity_y[orb]
        en = energije[orb] - mu
        fd1_ = fd1[orb]
        for m in range(Ny):
            for n in range(Nx//2+1):

                if n in [0,Nx//2]: multiply = 1
                else: multiply = 2

                K0_x.append( multiply * fd1_[m,n] * velx[m,n]**2 )
                K0_y.append( multiply * fd1_[m,n] * vely[m,n]**2 )

                K1_x.append( multiply * en[m,n] * fd1_[m,n] * velx[m,n]**2 )
                K1_y.append( multiply * en[m,n] * fd1_[m,n] * vely[m,n]**2 )   

    K0_x = kahan_sum(K0_x) * 2 / Nk # factor 2 for spin
    K0_y = kahan_sum(K0_y) * 2 / Nk
    K1_x = kahan_sum(K1_x) * 2 / Nk
    K1_y = kahan_sum(K1_y) * 2 / Nk   
    return K0_x, K0_y, K1_x, K1_y

@njit
def spektralna_orb(omegas, mu, energije_k, Gamma):
    N_orb = len(energije_k)
    A = np.zeros((len(omegas), N_orb))
    for orb in range(N_orb):
        A[:,orb] = 1/np.pi * Gamma / ( (omegas - (energije_k[orb] - mu))**2 + Gamma**2 )
    return A

@njit(parallel=True, cache=True)
def phi_Kubo(mat1, mat2, epsilons, energije, Gamma, mu):
    Norb, Ny, Nx = energije.shape
    Nk = Ny*Nx
    Nw = len(epsilons)

    phi_temporary = np.zeros((Ny, Nw))

    for m in prange(Ny):
        phi_local = np.zeros(Nw)

        for n in range(Nx//2+1):

            if n in (0,Nx//2):
                multiply = 1
            else:
                multiply = 2

            A = spektralna_orb(epsilons, mu, energije[:,m,n], Gamma)
            for a in range(Norb):
                for b in range(Norb):
                    phi_local += multiply * (mat1[a,b,m,n] * A[:,b] * mat2[b,a,m,n] * A[:,a]).real

        phi_temporary[m,:] = phi_local

    phi = np.sum(phi_temporary, axis=0)
    return 2 * phi / Nk # factor 2 for spin

def input_data(Kymesh, Kxmesh, a, b, pos, kinetic, interaction):
    Ny, Nx = Kxmesh.shape
    geom = dict()
    geom["kinetic"] = kinetic
    geom["interaction"] = interaction
    geom["pos"] = pos

    phases_kin = np.zeros((len(kinetic), Ny, Nx), dtype=np.complex128)
    for l, line in enumerate(kinetic):
        x, y, orb1, orb2, t = line
        x, y = float(x), float(y)
        phases_kin[l] = np.exp(-1j*(Kxmesh * x * a + Kymesh * y * b) )
    phases_int = np.zeros((len(interaction), Ny, Nx), dtype=np.complex128)
    for l, line in enumerate(interaction):
        x, y, orb1, orb2 = line
        x, y = float(x), float(y)
        phases_int[l] = np.exp(-1j*(Kxmesh * x * a + Kymesh * y * b) )
    phases = {"kin" : phases_kin,
              "int" : phases_int}
    return geom, phases

def G_ffts(phases, Ny, Nx):
    L = len(phases["kin"])
    M = len(phases["int"])
    g_ffts_M4a1 = np.zeros((L, M, Ny, Nx), dtype=np.complex128)
    g_ffts_M4a2 = np.copy(g_ffts_M4a1)
    g_ffts_M4b1 = np.copy(g_ffts_M4a1)
    g_ffts_M4b2 = np.copy(g_ffts_M4a1)

    for l in range(L):
        for m in range(M):
            g = np.conj(phases["kin"][l]) * phases["int"][m]
            g_ffts_M4a1[l, m] = np.fft.fft2(np.fft.ifftshift(g))

            g = phases["kin"][l] * np.conj(phases["int"][m])
            g_ffts_M4b1[l, m] = np.fft.fft2(np.fft.ifftshift(g))

            g = phases["int"][m]
            g_ffts_M4a2[l, m] = np.fft.fft2(np.fft.ifftshift(g))

            g = np.conj(phases["int"][m])
            g_ffts_M4b2[l, m] = np.fft.fft2(np.fft.ifftshift(g))
    return g_ffts_M4a1, g_ffts_M4a2, g_ffts_M4b1, g_ffts_M4b2

def compute_g_fft_single(phases, l, m, variant):
    """
    Compute a single (Ny,Nx) g_fft for given l,m and variant in
    ['M4a1', 'M4a2', 'M4b1', 'M4b2'].
    No large array stored.
    """
    if variant == 'M4a1':
        g = np.conj(phases["kin"][l]) * phases["int"][m]
    elif variant == 'M4b1':
        g = phases["kin"][l] * np.conj(phases["int"][m])
    elif variant == 'M4a2':
        g = phases["int"][m]
    elif variant == 'M4b2':
        g = np.conj(phases["int"][m])
    return np.fft.fft2(np.fft.ifftshift(g))

def compute_all_mf_matrices(Kymesh, rho, geom, phases, a, b, U, V):
    # note: g_ffts argument is GONE, computed on the fly instead
    Ny, Nx = Kymesh.shape
    Nk = Ny * Nx

    M3  = np.zeros((2, 6, 6, Ny, Nx), dtype=np.complex64)
    M6  = np.zeros((2, 6, 6, Ny, Nx), dtype=np.complex64)
    M4a = np.zeros((2, 6, 6, Ny, Nx), dtype=np.complex64)
    M4b = np.zeros((2, 6, 6, Ny, Nx), dtype=np.complex64)

    ns = np.array([np.sum(rho[i, i]).real for i in range(6)])

    rho_fft_cache  = {}   # key -> (Ny,Nx) fft array
    #g_fft_cache    = {}   # (l, m, variant) -> (Ny,Nx) fft array
    # peak memory for g_fft_cache: at most 4*L*(Ny*Nx) if all l,m visited
    # but realistically fills to 4*L*M*(Ny*Nx) only after full loop —
    # if that is still too much, set a max cache size (see below)

    # If each (l,m) is visited only once, caching buys nothing — just compute directly:
    def get_g_fft(l, m, variant):
        return compute_g_fft_single(phases, l, m, variant)

    def get_rho_fft(i, j):
        key = (i, j)
        if key not in rho_fft_cache:
            rho_fft_cache[key] = np.fft.fft2(np.fft.ifftshift(rho[i, j]))
        return rho_fft_cache[key]

    for l, (x, y, orb1, orb2, t) in enumerate(geom["kinetic"]):
        orb1, orb2 = int(orb1), int(orb2)
        phase_k = phases["kin"][l]
        fk = t * phase_k / Nk

        for m, (x_, y_, orb1_, orb2_) in enumerate(geom["interaction"]):
            orb1_, orb2_ = int(orb1_), int(orb2_)

            if orb2 == orb2_:
                lega = geom["pos"][orb2] - geom["pos"][orb1_] - np.array([x_*a, y_*b])
                if orb1_ == orb2_: V_ = U
                else: V_ = 2. * V

                for nu in range(2):
                    M3[nu, orb1-1, orb2-1] += -1j * t * V_ * lega[nu] * phase_k * ns[orb1_-1] / Nk

                suma = np.sum(rho[orb2-1, orb1-1] * phase_k)
                for nu in range(2):
                    M6[nu, orb1_-1, orb1_-1] += -1j * t * V_ * lega[nu] * suma / Nk

                if orb1_ == orb2_: V_ = U
                else: V_ = V

                rho_fft = get_rho_fft(orb1_-1, orb1-1)
                for nu in range(2):
                    g_fft = -1j * V_ * lega[nu] * get_g_fft(l, m, 'M4a1')
                    gh = np.fft.fftshift(np.fft.ifft2(g_fft * rho_fft))
                    M4a[nu, orb1_-1, orb2-1] += fk * gh

                h_fft = np.fft.fft2(np.fft.ifftshift(fk * rho[orb2-1, orb1_-1]))
                for nu in range(2):
                    g_fft = -1j * V_ * lega[nu] * get_g_fft(l, m, 'M4b1')
                    gh = np.fft.fftshift(np.fft.ifft2(g_fft * h_fft))
                    M4b[nu, orb1-1, orb1_-1] += gh

            if orb1 == orb2_:
                lega = geom["pos"][orb1] - geom["pos"][orb1_] - np.array([x_*a, y_*b])
                if orb1_ == orb2_: V_ = U
                else: V_ = 2. * V

                for nu in range(2):
                    M3[nu, orb1-1, orb2-1] += +1j * t * V_ * lega[nu] * phase_k * ns[orb1_-1] / Nk

                suma = np.sum(rho[orb2-1, orb1-1] * phase_k)
                for nu in range(2):
                    M6[nu, orb1_-1, orb1_-1] += +1j * t * V_ * lega[nu] * suma / Nk

                if orb1_ == orb2_: V_ = U
                else: V_ = V

                rho_fft = get_rho_fft(orb1_-1, orb1-1)
                for nu in range(2):
                    g_fft = +1j * V_ * lega[nu] * get_g_fft(l, m, 'M4a2')
                    gh = np.fft.fftshift(np.fft.ifft2(g_fft * rho_fft))
                    M4a[nu, orb1_-1, orb2-1] += fk * gh

                h_fft = np.fft.fft2(np.fft.ifftshift(fk * rho[orb2-1, orb1_-1]))
                for nu in range(2):
                    g_fft = +1j * V_ * lega[nu] * get_g_fft(l, m, 'M4b2')
                    gh = np.fft.fftshift(np.fft.ifft2(g_fft * h_fft))
                    M4b[nu, orb1-1, orb1_-1] += gh

    return 0.5*M3, 0.5*M6, -0.5*M4a, -0.5*M4b

''' this function does same as mf_matrix1,2,3,4
but it is more convenient if called many times because it uses some precomputed stuff '''
def compute_all_mf_matrices_old(Kymesh, rho, geom, phases, g_ffts, a, b, U, V):
    Ny, Nx = Kymesh.shape
    Nk = Ny * Nx

    M3 = np.zeros((2,6,6,Ny,Nx), dtype=np.complex64)
    M6 = np.zeros((2,6,6,Ny,Nx), dtype=np.complex64)
    M4a = np.zeros((2,6,6,Ny,Nx), dtype=np.complex64)
    M4b = np.zeros((2,6,6,Ny,Nx), dtype=np.complex64)

    ns = np.zeros(6)
    for i in range(6):
        ns[i] = np.sum(rho[i,i]).real

    g_ffts_M4a1, g_ffts_M4a2, g_ffts_M4b1, g_ffts_M4b2 = g_ffts

    rho_fft_cache = {}

    for l, (x, y, orb1, orb2, t) in enumerate(geom["kinetic"]):
        rho_fft_cache.clear()
        orb1, orb2 = int(orb1), int(orb2)
        phase_k = phases["kin"][l]
        fk = t * phase_k / Nk

        for m, (x_, y_, orb1_, orb2_) in enumerate(geom["interaction"]):
            orb1_, orb2_ = int(orb1_), int(orb2_)
            if orb2 == orb2_:
                lega = geom["pos"][orb2] - geom["pos"][orb1_] - np.array([x_*a, y_*b])
                if orb1_ == orb2_: V_ = U
                else: V_ = 2. * V

                # ---------- M3 ----------
                for nu in range(2):
                    M3[nu,orb1-1,orb2-1] += -1j * t * V_ * lega[nu] * phase_k * ns[orb1_-1] / Nk

                # ---------- M6 ----------
                suma = np.sum(rho[orb2-1,orb1-1] * phase_k)
                for nu in range(2):
                    M6[nu,orb1_-1,orb1_-1] += -1j * t * V_ * lega[nu] * suma / Nk

                if orb1_ == orb2_: V_ = U
                else: V_ = V

                # ---------- M4a ----------
                key = (orb1_-1, orb1-1)
                if key not in rho_fft_cache:
                    rho_fft_cache[key] = np.fft.fft2(
                        np.fft.ifftshift(rho[key])
                    )
                rho_fft = rho_fft_cache[key]
                for nu in range(2):
                    g_fft = -1j * V_ * lega[nu] * g_ffts_M4a1[l,m]
                    gh = np.fft.fftshift(
                        np.fft.ifft2(g_fft * rho_fft)
                    )
                    M4a[nu,orb1_-1,orb2-1] += fk * gh

                # ---------- M4b ----------
                h = fk * rho[orb2-1,orb1_-1]
                h_fft = np.fft.fft2(np.fft.ifftshift(h))
                for nu in range(2):
                    g_fft = -1j * V_ * lega[nu] * g_ffts_M4b1[l,m]
                    gh = np.fft.fftshift(np.fft.ifft2(g_fft * h_fft))
                    M4b[nu,orb1-1,orb1_-1] += gh

            if orb1 == orb2_:
                lega = geom["pos"][orb1] - geom["pos"][orb1_] - np.array([x_*a, y_*b])

                if orb1_ == orb2_: V_ = U
                else: V_ = 2. * V

                # ---------- M3 ----------
                for nu in range(2):
                    M3[nu,orb1-1,orb2-1] += +1j * t * V_ * lega[nu] * phase_k * ns[orb1_-1] / Nk

                # ---------- M6 ----------
                suma = np.sum(rho[orb2-1,orb1-1] * phase_k)
                for nu in range(2):
                    M6[nu,orb1_-1,orb1_-1] += +1j * t * V_ * lega[nu] * suma / Nk

                if orb1_ == orb2_: V_ = U
                else: V_ = V

                # ---------- M4a ----------
                key = (orb1_-1, orb1-1)
                if key not in rho_fft_cache:
                    rho_fft_cache[key] = np.fft.fft2(
                        np.fft.ifftshift(rho[key])
                    )
                rho_fft = rho_fft_cache[key]
                for nu in range(2):
                    g_fft = +1j * V_ * lega[nu] * g_ffts_M4a2[l,m]
                    gh = np.fft.fftshift(
                        np.fft.ifft2(g_fft * rho_fft)
                    )
                    M4a[nu,orb1_-1,orb2-1] += fk * gh
                
                # ---------- M4b ----------
                h = fk * rho[orb2-1,orb1_-1]
                h_fft = np.fft.fft2(np.fft.ifftshift(h))
                for nu in range(2):
                    g_fft = +1j * V_ * lega[nu] * g_ffts_M4b2[l,m]
                    gh = np.fft.fftshift(
                        np.fft.ifft2(g_fft * h_fft)
                    )
                    M4b[nu,orb1-1,orb1_-1] += gh
    return 0.5*M3, 0.5*M6, -0.5*M4a, -0.5*M4b

def mf_matrix1(Kymesh, Kxmesh, rho, a, b, U, V, pos, kinetic, interaction):
    Ny, Nx = Kxmesh.shape
    Nk = Ny * Nx
    matrix = np.zeros((2,6,6,Ny,Nx), dtype=np.complex128)

    for alpha in range(1,7):
        for beta in range(1,7):
            for line in kinetic:
                x, y, orb1, orb2, t = line
                x, y, orb1, orb2, t = float(x), float(y), int(orb1), int(orb2), float(t)
                if orb1 == alpha and orb2 == beta:
                    if orb1 == orb2 and x == 0: pass
                    for line_ in interaction:
                        x_, y_, orb1_, orb2_ = line_
                        x_, y_, orb1_, orb2_ = float(x_), float(y_), int(orb1_), int(orb2_)
                        if orb1_ == orb2_: V_ = U
                        else: V_ = 2 * V # factor 2 for spin multiplicity..
                        if orb2 == orb2_:
                            suma_n = np.sum(rho[orb1_-1, orb1_-1])
                            lega = pos[orb2] - pos[orb1_] - np.array([x_*a, y_*b])
                            for nu in range(2):
                                matrix[nu, orb1 - 1, orb2 - 1] += -1j * t * V_ * lega[nu] * np.exp(-1j * Kxmesh * x * a -1j * Kymesh * y * b) / Nk * suma_n

                        if orb1 == orb2_:
                            suma_n = np.sum(rho[orb1_-1, orb1_-1])
                            lega = pos[orb1] - pos[orb1_] - np.array([x_*a, y_*b])
                            for nu in range(2):
                                matrix[nu, orb1 - 1, orb2 - 1] += 1j * t * V_ * lega[nu] * np.exp(-1j * Kxmesh * x * a -1j * Kymesh * y * b) / Nk * suma_n
    return matrix * 0.5

def mf_matrix2(Kymesh, Kxmesh, rho, a, b, U, V, pos, kinetic, interaction):
    Ny, nx = Kxmesh.shape
    Nx = 2 * nx - 2
    Nk = Ny * Nx
    matrix = np.zeros((2, 6, 6, Ny, nx), dtype=np.complex128)

    for alpha in range(1,7):
        for beta in range(1,7):
            for line in kinetic:
                x, y, orb1, orb2, t = line
                x, y, orb1, orb2, t = float(x), float(y), int(orb1), int(orb2), float(t)
                if orb1 == alpha and orb2 == beta:
                    if orb1 == orb2 and x == 0: pass
                    for line_ in interaction:
                        x_, y_, orb1_, orb2_ = line_
                        x_, y_, orb1_, orb2_ = float(x_), float(y_), int(orb1_), int(orb2_)
                        if orb1_ == orb2_: V_ = U
                        else: V_ = 2 * V
                        if orb2 == orb2_:
                            rho_r = rho[orb2-1,orb1-1]
                            factor = np.exp(-1j * Kxmesh * x * a - 1j * Kymesh * y * b)
                            suma_n = np.sum(rho_r * factor)
                            lega = pos[orb2] - pos[orb1_] - np.array([x_*a, y_*b])
                            for nu in range(2):
                                matrix[nu, orb1_ -1, orb1_ - 1] += -1j * t * V_ * lega[nu] / Nk * suma_n

                        if orb1 == orb2_:
                            rho_r = rho[orb2-1,orb1-1]
                            factor = np.exp(-1j * Kxmesh * x * a - 1j * Kymesh * y * b)
                            suma_n = np.sum(rho_r * factor)
                            lega = pos[orb1] - pos[orb1_] - np.array([x_*a, y_*b])
                            for nu in range(2):
                                matrix[nu, orb1_ - 1, orb1_ - 1] += 1j * t * V_ * lega[nu]  / Nk * suma_n
    return matrix * 0.5 

# I verified that convolution via FFT yields the same as by direct sum; this is M^6 in my notes
def mf_matrix3(Kymesh, Kxmesh, rho, a, b, U, V, pos, kinetic, interaction):
    Ny, Nx = Kxmesh.shape
    Nk = Ny * Nx
    matrix = np.zeros((2, 6, 6, Ny, Nx), dtype=np.complex128)

    for alpha in range(1,7):
        for beta in range(1,7):
            for line in kinetic:
                x, y, orb1, orb2, t = line
                x, y, orb1, orb2, t = float(x), float(y), int(orb1), int(orb2), float(t)
                if orb1 == alpha and orb2 == beta:
                    if orb1 == orb2 and x == 0: pass
                    f_k = t * np.exp(-1j * Kxmesh * x * a - 1j * Kymesh * y * b) / Nk

                    for line_ in interaction:
                        x_, y_, orb1_, orb2_ = line_
                        x_, y_, orb1_, orb2_ = float(x_), float(y_), int(orb1_), int(orb2_)
                        if orb1_ == orb2_: pass 
                        if orb2 == orb2_:
                            lega = pos[orb2] - pos[orb1_] - np.array([x_*a, y_*b])
                            for nu in range(2):
                                g = -1j * V * lega[nu] * np.exp(1j * Kxmesh * x * a + 1j * Kymesh * y * b) * np.exp(-1j * Kxmesh * x_ * a - 1j * Kymesh * y_ * b)
                                h = rho[orb1_ - 1, orb1 - 1, :, :]

                                g_fft = np.fft.fft2(np.fft.ifftshift(g))
                                h_fft = np.fft.fft2(np.fft.ifftshift(h))

                                gh = np.fft.ifft2(g_fft * h_fft)
                                gh = np.fft.fftshift(gh)

                                matrix[nu, orb1_ -1, orb2 -1] +=  f_k * gh

                        if orb1 == orb2_:
                            lega = pos[orb1] - pos[orb1_] - np.array([x_*a, y_*b])
                            for nu in range(2):
                                g = 1j * V * lega[nu] * np.exp(-1j*Kxmesh*x_*a - 1j*Kymesh*y_*b)
                                h = rho[orb1_ - 1, orb1 - 1, :, :]

                                g_fft = np.fft.fft2(np.fft.ifftshift(g))
                                h_fft = np.fft.fft2(np.fft.ifftshift(h))

                                gh = np.fft.ifft2(g_fft * h_fft)
                                gh = np.fft.fftshift(gh)

                                matrix[nu, orb1_ -1, orb2 - 1] += f_k * gh
    return -matrix * 0.5 

# I verified that convolution via FFT yields the same as by direct sum; this is M^6 in my notes
def mf_matrix4(Kymesh, Kxmesh, rho, a, b, U, V, pos, kinetic, interaction):
    Ny, Nx = Kxmesh.shape
    Nk = Ny * Nx
    matrix = np.zeros((2, 6, 6, Ny, Nx), dtype=np.complex128)

    for alpha in range(1,7):
        for beta in range(1,7):
            for line in kinetic:
                x, y, orb1, orb2, t = line
                x, y, orb1, orb2, t = float(x), float(y), int(orb1), int(orb2), float(t)
                if orb1 == alpha and orb2 == beta:
                    if orb1 == orb2 and x == 0: pass
                    for line_ in interaction:
                        x_, y_, orb1_, orb2_ = line_
                        x_, y_, orb1_, orb2_ = float(x_), float(y_), int(orb1_), int(orb2_)
                        if orb1_ == orb2_: pass 
                        if orb2 == orb2_:
                            lega = pos[orb2] - pos[orb1_] - np.array([x_*a, y_*b])
                            for nu in range(2):
                                g = -1j * V * lega[nu] * np.exp(-1j * Kxmesh * x * a - 1j * Kymesh * y * b) * np.exp(1j * Kxmesh * x_ * a + 1j * Kymesh * y_ * b)
                                h = t * np.exp(-1j * Kxmesh * x * a - 1j * Kymesh * y * b) * rho[orb2 - 1, orb1_ - 1, :, :] / Nk

                                g_fft = np.fft.fft2(np.fft.ifftshift(g))
                                h_fft = np.fft.fft2(np.fft.ifftshift(h))

                                gh = np.fft.ifft2(g_fft * h_fft)
                                gh = np.fft.fftshift(gh)

                                matrix[nu, orb1 - 1, orb1_ -1] +=  gh
                        if orb1 == orb2_:
                            lega = pos[orb1] - pos[orb1_] - np.array([x_*a, y_*b])
                            for nu in range(2):
                                g = 1j * V * lega[nu] * np.exp(1j * Kxmesh * x_ * a +  1j * Kymesh * y_ * b)
                                h = t * np.exp(-1j * Kxmesh * x * a - 1j * Kymesh * y * b) * rho[orb2 - 1, orb1_ - 1, :, :] / Nk

                                g_fft = np.fft.fft2(np.fft.ifftshift(g))
                                h_fft = np.fft.fft2(np.fft.ifftshift(h))
                                gh = np.fft.ifft2(g_fft * h_fft)
                                gh = np.fft.fftshift(gh)
                                matrix[nu, orb1 - 1, orb1_ - 1] += gh
    return -matrix * 0.5

def interaction_expand(interaction, U, V, a):
    storing = []
    thetas = []
    parities = []
    for i, (x, _, orb1, orb2) in enumerate(interaction):
        if orb1 == orb2:
            nus = [0]
            Vs = [U/4]
            delta = 0.
            thetas += Vs
            parities += [1]
            storing.append([orb1, orb2, delta, nus, Vs])
        elif orb1 < orb2:
            nus = [0,1,2,3]
            Vs = [V, -V/2, -V/2, -V]
            delta = a * x
            thetas += Vs
            parities += [1, 1, -1, 1]
            storing.append([orb1, orb2, delta, nus, Vs])
    return storing, thetas, parities


sigmas = np.zeros((4, 2, 2), dtype=np.complex128)
sigmas[0] = np.eye(2)
sigmas[1] = np.array([[0,1],[1,0]])
sigmas[2] = np.array([[0,-1j], [1j,0]])
sigmas[3] = np.diag([1,-1])

def rho_operators(Nop, Kymesh, Kxmesh, interaction, a, b):
    Ny, Nx = Kymesh.shape
    rhos = np.zeros((Nop, 6, 6, Ny, Nx), dtype=np.complex64)
    u = 0
    for l, (x, y, orb1, orb2) in enumerate(interaction):
        orb1, orb2 = int(orb1), int(orb2)
        if orb1 < orb2:
            P = np.zeros((6, 2))
            P[orb1-1,0] = 1
            P[orb2-1,1] = 1
            for nu in [0,1,2,3]:
                U_kdelta = np.zeros((2, 2, Ny, Nx), dtype=np.complex64)
                factor = np.exp(-1j * (Kxmesh * x * a + Kymesh * y * b) / 2)
                U_kdelta[0,0] = factor
                U_kdelta[1,1] = factor.conj()
                R = np.einsum('ij,jlxy-> ilxy', P, U_kdelta)
                rhos[u] = np.einsum('ijxy, jl, mlxy-> imxy', R, sigmas[nu], R.conj())
                u += 1
        if orb1 == orb2:
            P = np.zeros((6, 2))
            P[orb1-1,0] = 1
            P[orb2-1,1] = 1
            for nu in [0]:
                U_kdelta = np.zeros((2, 2, Ny, Nx), dtype=np.complex64)
                factor = np.exp(-1j * (Kxmesh * x * a + Kymesh * y * b) / 2)
                U_kdelta[0,0] = factor
                U_kdelta[1,1] = factor.conj()
                R = np.einsum('ij,jlxy-> ilxy', P, U_kdelta)
                rhos[u] = np.einsum('ijxy, jl, mlxy-> imxy', R, sigmas[nu], R.conj())
                u += 1
    return rhos

def make_single_rho_tilde(l_idx, interaction, a, b, Kymesh, Kxmesh, vecs):
    """
    Compute rhos_tilde for a SINGLE operator index l_idx.
    Returns an array of shape (6, 6, Ny, nx), dtype=complex64.
    """
    Ny, nx = Kymesh.shape
    # Walk through interaction list to find which (x,y,orb1,orb2,nu)
    # corresponds to flat index l_idx
    u = 0
    for x, y, orb1, orb2 in interaction:
        orb1, orb2 = int(orb1), int(orb2)
        nus = [0, 1, 2, 3] if orb1 < orb2 else ([0] if orb1 == orb2 else [])
        for nu in nus:
            if u == l_idx:
                # ---- build this single rho ----
                P = np.zeros((6, 2))
                P[orb1 - 1, 0] = 1
                P[orb2 - 1, 1] = 1
                factor = np.exp(
                    -1j * (Kxmesh * x * a + Kymesh * y * b) / 2
                ).astype(np.complex64)
                U_kdelta = np.zeros((2, 2, Ny, nx), dtype=np.complex64)
                U_kdelta[0, 0] = factor
                U_kdelta[1, 1] = factor.conj()
                R = np.einsum('ij,jlxy->ilxy', P, U_kdelta)
                rho = np.einsum('ijxy,jl,mlxy->imxy',
                                R, sigmas[nu], R.conj())
                # ---- apply operator_tilde transform ----
                return operator_tilde(rho, vecs)   # your existing function
            u += 1
    raise IndexError(f"l_idx={l_idx} out of range (Nop={u})")

def make_rho_tilde_factory(interaction, a, b, Kymesh, Kxmesh, vecs):
    """Returns a callable  f(i) -> rhos_tilde[i]  computed on the fly."""
    def _get(i):
        return make_single_rho_tilde(i, interaction, a, b,
                                     Kymesh, Kxmesh, vecs)
    return _get

''' Fermi-Dirac function '''
@njit
def fd(eps, mu, T):
    return 1.0 / (np.exp((eps - mu) / T) + 1.0)

@njit(cache=True)
def Pi_bubble_tilde(omega, E_mk, E_nk, Gamma, mu_, invt, nodes, weights, eps=1e-5, n_eps=1.0):
    w    = omega / Gamma
    e_mk = E_mk  / Gamma
    e_nk = E_nk  / Gamma
    
    T = Gamma / invt
    
    invpi = 1.0 / np.pi

    # Single, T-independent cutoff based on Lorentzian tail
    # A(e) ~ 1/(pi * e^2) < eps  =>  e > 1/(pi*eps)
    epsilon_max = np.sqrt(np.abs(np.arccosh(1/(eps*4*T))) * 2 * T) / Gamma * n_eps

    # Three integration intervals, one per peak
    centers = np.array([e_mk, e_nk - w, mu_])

    # Build, sort, merge intervals (same as your new code)
    raw = np.empty((3, 2), dtype=np.float64)
    for c in range(3):
        raw[c, 0] = centers[c] - epsilon_max
        raw[c, 1] = centers[c] + epsilon_max

    # Sort by left endpoint
    for i in range(3):
        for j in range(i + 1, 3):
            if raw[j, 0] < raw[i, 0]:
                raw[i, 0], raw[j, 0] = raw[j, 0], raw[i, 0]
                raw[i, 1], raw[j, 1] = raw[j, 1], raw[i, 1]

    # Merge overlapping intervals
    merged  = np.empty((3, 2), dtype=np.float64)
    merged[0, 0] = raw[0, 0]
    merged[0, 1] = raw[0, 1]
    n_merged = 1
    for i in range(1, 3):
        if raw[i, 0] <= merged[n_merged - 1, 1]:
            merged[n_merged - 1, 1] = max(raw[i, 1], merged[n_merged - 1, 1])
        else:
            merged[n_merged, 0] = raw[i, 0]
            merged[n_merged, 1] = raw[i, 1]
            n_merged += 1

    # Integrate over merged intervals
    n_nodes  = len(nodes)

    res_mn_r = 0.0; res_mn_i = 0.0
    res_nm_r = 0.0; res_nm_i = 0.0
    res_w_mn_r = 0.0; res_w_mn_i = 0.0
    res_w_nm_r = 0.0; res_w_nm_i = 0.0

    for s in range(n_merged):
        a    = merged[s, 0]
        b    = merged[s, 1]
        mid  = 0.5 * (a + b)
        half = 0.5 * (b - a)

        for i in range(n_nodes):
            e   = mid + half * nodes[i]
            ew  = e + w
            dm  = e - e_mk
            dn  = e - e_nk
            dmw = dm + w
            dnw = dn + w
            pref = e - mu_ + 0.5 * w
            wi   = weights[i] * half

            f  = 1.0 / (np.exp((e  - mu_) * invt) + 1.0)
            fw = 1.0 / (np.exp((ew - mu_) * invt) + 1.0)

            A_mk  = invpi / (dm  * dm  + 1.0)
            A_nk  = invpi / (dn  * dn  + 1.0)
            A_mkw = invpi / (dmw * dmw + 1.0)
            A_nkw = invpi / (dnw * dnw + 1.0)

            Grnw_r =  dnw / (dnw * dnw + 1.0)
            Grnw_i = -1.0 / (dnw * dnw + 1.0)
            Grmw_r =  dmw / (dmw * dmw + 1.0)
            Grmw_i = -1.0 / (dmw * dmw + 1.0)
            Gam_r  =  dm  / (dm  * dm  + 1.0)
            Gam_i  =  1.0 / (dm  * dm  + 1.0)
            Gan_r  =  dn  / (dn  * dn  + 1.0)
            Gan_i  =  1.0 / (dn  * dn  + 1.0)

            mn_r = A_mk * Grnw_r * f + A_nkw * Gam_r * fw
            mn_i = A_mk * Grnw_i * f + A_nkw * Gam_i * fw
            nm_r = A_nk * Grmw_r * f + A_mkw * Gan_r * fw
            nm_i = A_nk * Grmw_i * f + A_mkw * Gan_i * fw

            res_mn_r   += wi * mn_r
            res_mn_i   += wi * mn_i
            res_nm_r   += wi * nm_r
            res_nm_i   += wi * nm_i
            res_w_mn_r += wi * pref * mn_r
            res_w_mn_i += wi * pref * mn_i
            res_w_nm_r += wi * pref * nm_r
            res_w_nm_i += wi * pref * nm_i

    res_mn   = (res_mn_r   + 1j * res_mn_i)   / Gamma
    res_nm   = (res_nm_r   + 1j * res_nm_i)   / Gamma
    res_w_mn =  res_w_mn_r + 1j * res_w_mn_i
    res_w_nm =  res_w_nm_r + 1j * res_w_nm_i

    return res_mn, res_nm, res_w_mn, res_w_nm

@njit(parallel=True, cache=True)
def precompute_Pi_all(omega, energije, Gamma, mu_, invt, nodes, weights, eps=1e-5):
    Norb, Ny, Nx = energije.shape

    pi_mn  = np.zeros((Norb, Norb, Ny, Nx), dtype=np.complex128)
    pi_nm  = np.zeros((Norb, Norb, Ny, Nx), dtype=np.complex128)
    piw_mn = np.zeros((Norb, Norb, Ny, Nx), dtype=np.complex128)
    piw_nm = np.zeros((Norb, Norb, Ny, Nx), dtype=np.complex128)

    for i in prange(Ny):
        for j in range(Nx):
            for m in range(Norb):
                for n in range(m,Norb):

                    pi_mnk, pi_nmk, pie_mnk, pie_nmk = Pi_bubble_tilde(omega, energije[m,i,j], energije[n,i,j], Gamma, mu_, invt, nodes, weights, eps)

                    pi_mn[m,n,i,j] = pi_mnk
                    pi_nm[m,n,i,j] = pi_nmk
                    piw_mn[m,n,i,j] = pie_mnk
                    piw_nm[m,n,i,j] = pie_nmk

    return pi_mn, pi_nm, piw_mn, piw_nm

@njit(parallel=True, cache=True)
def chi_UV(U, V, pi_mn, pi_nm):
    Norb, Ny, Nx = U.shape[-3:]
    Nk = Ny * Nx
    chi = 0.0 + 0.0j

    for i in prange(Ny):
        chi_i = 0.0 + 0.0j

        for j in range(Nx):

            chi_ij = 0.0 + 0.0j

            for m in range(Norb):
                for n in range(m, Norb):

                    U_mn = U[m,n,i,j]
                    U_nm = U[n,m,i,j]
                    V_mn = V[m,n,i,j]
                    V_nm = V[n,m,i,j]

                    p_mn = pi_mn[m,n,i,j]
                    p_nm = pi_nm[m,n,i,j]

                    if m == n:
                        chi_ij += 0.5 * (U_mn * V_nm * p_nm + U_nm * V_mn * p_mn)
                    else:
                        chi_ij += U_mn * V_nm * p_nm + U_nm * V_mn * p_mn
            chi_i += chi_ij

        chi += chi_i
    return chi / Nk      

def get_rho_tilde(i, cache, factory, lock):
    if i not in cache:
        with lock:
            if i not in cache:
                cache[i] = factory(i)
    return cache[i]

def clear_rho_tilde_cache(rho_tilde_cache, rho_tilde_lock, cache=None, lock=None):
    if cache is None:
        cache = rho_tilde_cache
    if lock is None:
        lock = rho_tilde_lock
    with lock:
        cache.clear()

def compute_single_om_fused(
    om,
    Gamma, mu_, invt, nodes, weights,
    thetas, parities, tok_tilde_x, mat_tilde_x, tok_tilde_y, mat_tilde_y,
    energije,
    rho_tilde_factory, rho_tilde_cache, rho_tilde_lock,
    eps=1e-5
):
    Nop = len(thetas)
    
    thetas_diag = np.diag(thetas)
    I = np.eye(Nop)

    # ── ONE precomputation pass for this omega ──────────────────────────
    pi_mn, pi_nm, piw_mn, piw_nm = precompute_Pi_all(
        om, energije, Gamma, mu_, invt, nodes, weights, eps
    )

    # ── chi0 matrix  (Nop x Nop calls, but now cheap) ──────────────────
    chi0 = np.zeros((Nop, Nop), dtype=np.complex128)
    
    for i in range(Nop):
        eps_i = parities[i]
        rho_i = get_rho_tilde(i, rho_tilde_cache, rho_tilde_factory, rho_tilde_lock)
        for j in range(Nop):
            eps_j = parities[j]
            rho_j = get_rho_tilde(j, rho_tilde_cache, rho_tilde_factory, rho_tilde_lock)
            chi0[i, j] = chi_UV(rho_i, rho_j, pi_mn, pi_nm)
            chi0[j, i] = chi_UV(rho_j, rho_i, pi_mn, pi_nm)

    # ── chi_jj0 ────────────────────────────────────────────────────────
    chi_jj0_x = chi_UV(tok_tilde_x, tok_tilde_x, pi_mn, pi_nm)
    chi_jEj0_x = chi_UV(tok_tilde_x, tok_tilde_x, piw_mn, piw_nm)
    chi_matj0_x = chi_UV(mat_tilde_x, tok_tilde_x, pi_mn, pi_nm)

    chi_jj0_y = chi_UV(tok_tilde_y, tok_tilde_y, pi_mn, pi_nm)
    chi_jEj0_y = chi_UV(tok_tilde_y, tok_tilde_y, piw_mn, piw_nm)
    chi_matj0_y = chi_UV(mat_tilde_y, tok_tilde_y, pi_mn, pi_nm)

    # ── chi_jrho0 / chi_rhoj0 ──────────────────────────────────────────
    chi_jrho0_x = np.zeros(Nop, dtype=np.complex128)
    chi_rhoj0_x = np.zeros(Nop, dtype=np.complex128)
    chi_jErho0_x = np.zeros(Nop, dtype=np.complex128)
    chi_matrho0_x = np.zeros(Nop, dtype=np.complex128)
    chi_jrho0_y = np.zeros(Nop, dtype=np.complex128)
    chi_rhoj0_y = np.zeros(Nop, dtype=np.complex128)
    chi_jErho0_y = np.zeros(Nop, dtype=np.complex128)
    chi_matrho0_y = np.zeros(Nop, dtype=np.complex128)
    for i in range(Nop):
        eps_i = parities[i]
        rho_i = get_rho_tilde(i, rho_tilde_cache, rho_tilde_factory, rho_tilde_lock)
        chi_jrho0_x[i] = chi_UV(tok_tilde_x, rho_i, pi_mn, pi_nm)
        chi_rhoj0_x[i] = chi_UV(rho_i, tok_tilde_x, pi_mn, pi_nm)
        chi_jErho0_x[i] = chi_UV(tok_tilde_x, rho_i, piw_mn, piw_nm)
        chi_matrho0_x[i] = chi_UV(mat_tilde_x, rho_i, pi_mn, pi_nm)
        chi_jrho0_y[i] = chi_UV(tok_tilde_y, rho_i, pi_mn, pi_nm)
        chi_rhoj0_y[i] = chi_UV(rho_i, tok_tilde_y, pi_mn, pi_nm)
        chi_jErho0_y[i] = chi_UV(tok_tilde_y, rho_i, piw_mn, piw_nm)
        chi_matrho0_y[i] = chi_UV(mat_tilde_y, rho_i, pi_mn, pi_nm)

    # ── RPA ────────────────────────────────────────────────────────────
    mat     = I - chi0 @ thetas_diag
    inv     = LA.inv(mat)
    chi_rpa = inv @ chi0

    dchi_jj_x = chi_jrho0_x @ thetas_diag @ inv @ chi_rhoj0_x
    dchi_jEj_x = chi_jErho0_x @ thetas_diag @ inv @ chi_rhoj0_x
    dchi_matj_x = chi_matrho0_x @ thetas_diag @ inv @ chi_rhoj0_x
    dchi_jj_y = chi_jrho0_y @ thetas_diag @ inv @ chi_rhoj0_y
    dchi_jEj_y = chi_jErho0_y @ thetas_diag @ inv @ chi_rhoj0_y
    dchi_matj_y = chi_matrho0_y @ thetas_diag @ inv @ chi_rhoj0_y

    return om, chi0, chi_rpa, chi_jj0_x, dchi_jj_x, chi_jEj0_x, dchi_jEj_x, chi_matj0_x, dchi_matj_x, chi_jj0_y, dchi_jj_y, chi_jEj0_y, dchi_jEj_y, chi_matj0_y, dchi_matj_y

def compute_chi(
    omegas,
    Gamma, mu_, invt, nodes, weights,
    thetas, parities, tok_tilde_x, mat_tilde_x, tok_tilde_y, mat_tilde_y,
    energije,
    rho_tilde_factory,
    verbose=True,
    n_workers=None, #None: number of CPU cores, or specify an integer
    eps=1e-5
):
    omegas = np.asarray(omegas)
    N_om   = len(omegas)
    Nop    = len(thetas)

    rho_tilde_cache = {}
    rho_tilde_lock = threading.Lock()

    def _worker(om_idx, om):
        result = compute_single_om_fused(
            om,
            Gamma, mu_, invt, nodes, weights,
            thetas, parities, tok_tilde_x, mat_tilde_x, tok_tilde_y, mat_tilde_y,
            energije,
            rho_tilde_factory,
            rho_tilde_cache,   # <-- shared, persistent
            rho_tilde_lock,
            eps=eps
        )
        return om_idx, result

    chi0_arr      = np.zeros((N_om, Nop, Nop), dtype=np.complex128)
    chi_rpa_arr   = np.zeros((N_om, Nop, Nop), dtype=np.complex128)
    chi_jj0_arr_x   = np.zeros(N_om,             dtype=np.complex128)
    dchi_jj_arr_x   = np.zeros(N_om,             dtype=np.complex128)
    chi_matj0_arr_x = np.zeros(N_om, dtype=np.complex128)
    chi_jEj0_arr_x = np.zeros(N_om, dtype=np.complex128)
    dchi_matj_arr_x = np.zeros(N_om, dtype=np.complex128)
    dchi_jEj_arr_x = np.zeros(N_om, dtype=np.complex128)
    chi_jj0_arr_y   = np.zeros(N_om,             dtype=np.complex128)
    dchi_jj_arr_y   = np.zeros(N_om,             dtype=np.complex128)
    chi_matj0_arr_y = np.zeros(N_om, dtype=np.complex128)
    chi_jEj0_arr_y = np.zeros(N_om, dtype=np.complex128)
    dchi_matj_arr_y = np.zeros(N_om, dtype=np.complex128)
    dchi_jEj_arr_y = np.zeros(N_om, dtype=np.complex128)

    t_total = time.time()

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_worker, om_idx, om): om_idx
            for om_idx, om in enumerate(omegas)
        }

        with tqdm(total=N_om, desc="Progress:", disable=not verbose) as pbar:
            for future in as_completed(futures):
                om_idx, result = future.result()
                om, chi0, chi_rpa, chi_jj0_x, dchi_jj_x, chi_jEj0_x, dchi_jEj_x, chi_matj0_x, dchi_matj_x, chi_jj0_y, dchi_jj_y, chi_jEj0_y, dchi_jEj_y, chi_matj0_y, dchi_matj_y = result

                chi0_arr[om_idx]      = chi0
                chi_rpa_arr[om_idx]   = chi_rpa
                chi_jj0_arr_x[om_idx]   = chi_jj0_x
                dchi_jj_arr_x[om_idx]   = dchi_jj_x
                chi_jEj0_arr_x[om_idx] = chi_jEj0_x
                dchi_jEj_arr_x[om_idx] = dchi_jEj_x
                chi_matj0_arr_x[om_idx] = chi_matj0_x
                dchi_matj_arr_x[om_idx] = dchi_matj_x
                chi_jj0_arr_y[om_idx]   = chi_jj0_y
                dchi_jj_arr_y[om_idx]   = dchi_jj_y
                chi_jEj0_arr_y[om_idx] = chi_jEj0_y
                dchi_jEj_arr_y[om_idx] = dchi_jEj_y
                chi_matj0_arr_y[om_idx] = chi_matj0_y
                dchi_matj_arr_y[om_idx] = dchi_matj_y
                
                pbar.update(1)

    results_x = {'chi0' : chi0_arr,
               'chi' : chi_rpa_arr,

               'chi_jj0' : chi_jj0_arr_x,
               'dchi_jj' : dchi_jj_arr_x,

               'chi_jEj0' : chi_jEj0_arr_x,
               'chi_matj0' : chi_matj0_arr_x,

               'dchi_matj' : dchi_matj_arr_x,
               'dchi_jEj' : dchi_jEj_arr_x}
    
    results_y = {'chi0' : chi0_arr,
               'chi' : chi_rpa_arr,

               'chi_jj0' : chi_jj0_arr_y,
               'dchi_jj' : dchi_jj_arr_y,

               'chi_jEj0' : chi_jEj0_arr_y,
               'chi_matj0' : chi_matj0_arr_y,

               'dchi_matj' : dchi_matj_arr_y,
               'dchi_jEj' : dchi_jEj_arr_y}
    
    return results_x, results_y

''' operator in band basis obtained from operator in orbital basis '''
def operator_tilde(op_bare, vecs):
    op_tilde = np.empty_like(op_bare, dtype=np.complex128)
    if len(op_tilde.shape) > 4:
        # this means there are multiple operators
        n_ops = op_tilde.shape[0]
        for n in range(n_ops):
            op_tilde[n] = np.einsum('jixy, jlxy, lmxy -> imxy', vecs.conj(), op_bare[n], vecs)
    else:
        op_tilde = np.einsum('jixy, jlxy, lmxy -> imxy', vecs.conj(), op_bare, vecs)
    return op_tilde

def find_flat_regime(omegas, chi_omega, window=10):
    """
    Find flattest window in dchi_domega, using sliding window in LOG omega space.
    Relative std (std/|mean|) is minimized in the flat region.
    """
    dchi_domega = np.gradient(chi_omega, omegas)
    n          = len(omegas)
    rel_std    = np.full(n, np.nan)

    for i in range(n - window):
        chunk      = dchi_domega[i : i + window]
        mean       = np.mean(chunk)
        std        = np.std(chunk)
        rel_std[i] = std / np.abs(mean)

    # Best window = minimum relative std
    best_start = np.nanargmin(rel_std)
    best_end   = best_start + window

    # Expand window outward as long as rel_std stays low
    threshold = rel_std[best_start] * 3   # allow 3x the minimum rel_std
    
    # expand left
    left = best_start
    while left > 0 and rel_std[left - 1] < threshold:
        left -= 1
    
    # expand right
    right = best_end
    while right < n - window and rel_std[right] < threshold:
        right += 1

    return left, right

def get_dc_coefficient(omegas, chi_imag, omega_cutoff=None):
    """
    Get DC coefficient (alpha = chi_imag / omega as omega -> 0)
    for log-spaced omega arrays using weighted linear regression.
    
    Fits: chi_imag = alpha * omega + beta * omega^3
    Weights = 1/omega to ensure equal contribution per decade.
    """
    
    # Select low-frequency window
    if omega_cutoff is None:
        log_min = np.log10(omegas.min())
        log_max = np.log10(omegas.max())
        omega_cutoff = 10 ** (log_min + 0.2 * (log_max - log_min))
    
    mask = omegas <= omega_cutoff
    w    = 1.0 / omegas[mask]          # weights: uniform per decade
    x    = omegas[mask]
    y    = chi_imag[mask]

    # Weighted least squares: chi_imag = alpha * omega + beta * omega^3
    # Design matrix
    A  = np.column_stack([x, x**3])
    Aw = A * w[:, None]                # apply weights to rows
    yw = y * w

    # Solve weighted normal equations
    coeffs, _, _, _ = np.linalg.lstsq(Aw, yw, rcond=None)
    alpha, beta = coeffs

    return alpha, beta

# ====== response and susceptibility obtained from simulation of a pulse ======

''' Gaussian pulse modulated by a cosine (in practice, however, I choose Omega=0, i.e. the pulse is a Gaussian ''' 
def A_pulz(t, A0, t0, sigma, Omega):
    return A0 * np.cos(Omega * t) * np.exp(-(t-t0)**2/(2*sigma**2))
@njit
def relax_rho(rho, rho_eq, dt, Gamma):
    decay = np.exp(-Gamma * dt)
    return rho_eq + decay * (rho - rho_eq)

''' expectation value of measure_operators when system is described by density matrix rho'''
@njit(parallel=True)
def measure(measure_operators, rho):
    # Ensure we handle the dimensions correctly
    # rho is (dim, dim, Ny, Nx)
    # measure_operators should be (Nop, dim, dim, Ny, Nx)
    dim, _, Ny, Nx = rho.shape
    Nop = measure_operators.shape[0]
    
    measurements_k = np.zeros((Nop, Ny, Nx), dtype=np.complex128)
    
    for m in prange(Ny):
        for n in range(Nx):
            # Local slice of density matrix
            rho_mn = rho[:, :, m, n]
            
            for u in range(Nop):
                # Now indexing exactly 5 dimensions
                op_mn = measure_operators[u, :, :, m, n]
                
                # Manual trace: Tr(rho * op) = sum_{i,j} rho_{ij} * op_{ji}
                val = 0.0 + 0.0j
                for i in range(dim):
                    for j in range(dim):
                        val += rho_mn[i, j] * op_mn[j, i]
                measurements_k[u, m, n] = val
                
    # Sum over k-space dimensions (Ny, Nx)
    measurements = np.zeros(Nop, dtype=np.complex128)
    for u in range(Nop):
        measurements[u] = np.sum(measurements_k[u])
        
    return measurements


def build_measure_operators(measure_providers, Kymesh, rho, geom, phases, g_ffts, a, b, U, V):
    ops = []
    for provider in measure_providers:
        if callable(provider):
            op1, op2, op3, op4 = provider(Kymesh, rho, geom, phases, g_ffts, a, b, U, V)
            op = op1 + op2 + op3 + op4
        else:
            op = provider
        if op.ndim == 3:
            op = op[np.newaxis, ...]
        ops.append(op)
    return np.concatenate(ops, axis=0)


def evolve_chunk(data_tuple):
    """
    Worker function to evolve a subset of k-points.
    data_tuple contains (H_chunk, rho_chunk, dt)
    """
    H_chunk, rho_chunk, dt = data_tuple
    n_k = H_chunk.shape[0]
    dim = H_chunk.shape[1]
    
    rho_next_chunk = np.empty_like(rho_chunk)
    
    for i in range(n_k):
        # 1. Diagonalize the Hamiltonian slice
        # Using eigh is safe and fast here
        w, v = np.linalg.eigh(H_chunk[i])
        
        # 2. Rotate rho into the eigenbasis: rho_eig = V^H @ rho @ V
        rho_eig = v.conj().T @ rho_chunk[i] @ v
        
        # 3. Apply phase evolution: rho_ab * exp(-i(Ea - Eb)dt)
        # We can use broadcasting for speed:
        # Construct a matrix of (Ea - Eb)
        energy_diff = w[:, None] - w[None, :]
        phase_matrix = np.exp(-1j * energy_diff * dt)
        rho_eig *= phase_matrix
        
        # 4. Rotate back to original basis: rho_next = V @ rho_eig @ V^H
        rho_next_chunk[i] = v @ rho_eig @ v.conj().T
        
    return rho_next_chunk

import time
from concurrent.futures import ProcessPoolExecutor
# Crucial: Prevent nested thread contention
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

def parallel_density_evolution(hamiltonian, rho, dt, executor, n_workers=8):
    """Evolves rho using a persistent executor to avoid process-spawning overhead."""
    h_chunks = np.array_split(hamiltonian, n_workers)
    rho_chunks = np.array_split(rho, n_workers)
    
    tasks = [(h_chunks[i], rho_chunks[i], dt) for i in range(n_workers)]
    results = list(executor.map(evolve_chunk, tasks))
    
    return np.concatenate(results)

def H_full(Ny, Nx, hop, fock, hartree):
    H = hop + fock
    for i in range(Ny):
        for j in range(Nx):
            H[:,:,i,j] += hartree
    return H

''' main function which propagates the system and measures observables (measure_provider)
upon application of a perturbation (generated by perturbation_operator)
* do_freeze=True means that Hartree-Fock is frozen to its equilibrium value, hence we observe no corrections, e.g., in current-current response
* do_freeze=False is the opposite; Hartree-Fock is dynamic, i.e. densities respond to perturbations, and in this response the vertex corrections are captured
'''
def simulate_pulz(Kymesh, Kxmesh, hop, rho, a, b, U, V, perturbation_operator, measure_provider,
                  A0, t0_pulse, sigma, Omega, dt, t_max, Gamma,
                  do_freeze, Ncorr, tol, geom, phases, g_ffts, show_print=True,
                  hartree_list=None):
    
    N_points = int(t_max/dt)
    Ny, Nx = Kymesh.shape
    Nk = Ny * Nx
    rho_eq = np.copy(rho)
    
    # Initialize timing and operators
    t_fock = t_hartree = t_evolve = t_measure = 0.0
    measure_operators_raw = build_measure_operators(measure_provider, Kymesh, rho, geom, phases, g_ffts, a, b, U, V)

    # Force it to be 5D: (Nop, dim, dim, Ny, Nx)
    if measure_operators_raw.ndim == 4:
        # Add the missing 'u' dimension at index 0
        measure_operators = measure_operators_raw[np.newaxis, :, :, :, :]
    else:
        measure_operators = measure_operators_raw

    measure_operators = np.ascontiguousarray(measure_operators, dtype=np.complex128)
    
    fock_eq = helpers.H_fock(Kxmesh, Nk, rho, a, V)
    hartree_eq = helpers.H_hartree(rho, Nk, U, V, hartree_list)

    f_max = np.max(np.abs(fock_eq))
    h_max = np.max(np.abs(hartree_eq))

    rho_expvals = np.zeros((N_points, measure_operators.shape[0]), dtype=np.complex128)
    
    # Start the persistent process pool
    with ProcessPoolExecutor(max_workers=8) as executor:
        for i in range(N_points):
            t_now = i * dt
            if show_print and i % 50 == 0:
                print(f"Progress: {i/N_points:.1%}", flush=True)

            A_t = A_pulz(t_now, A0, t0_pulse, sigma, Omega)
            A_half = A_pulz(t_now + dt/2, A0, t0_pulse, sigma, Omega)

            # The state at the start of the timestep
            rho_start = np.copy(rho)
            rho_guess = np.copy(rho)

            for corr_step in range(Ncorr):
                # 1. Obtain Mean Fields
                if do_freeze:
                    f0, h0 = fock_eq, hartree_eq
                    f1, h1 = fock_eq, hartree_eq
                else:
                    # Predictor stage Hamiltonian
                    start = time.perf_counter()
                    f0 = helpers.H_fock(Kxmesh, Nk, rho_guess, a, V)
                    t_fock += time.perf_counter() - start
                    
                    start = time.perf_counter()
                    h0 = helpers.H_hartree(rho_guess, Nk, U, V, hartree_list)
                    t_hartree += time.perf_counter() - start
                    
                    # For the corrector, we need the HF fields at the GUESSED next step
                    # We'll calculate f1, h1 after the first evolution
                
                # 2. Evolution Logic
                start = time.perf_counter()
                
                # Apply initial relaxation if Gamma > 0
                rho_to_evolve = relax_rho(rho_start, rho_eq, dt/2, Gamma) if Gamma != 0 else rho_start
                
                # Determine Hamiltonian for this iteration
                if corr_step == 0:
                    # Predictor: Use H(t)
                    H_eff = H_full(Ny, Nx, hop - A_t * perturbation_operator, f0, h0)
                else:
                    # Corrector: Use average H
                    H_eff = H_full(Ny, Nx, hop - A_half * perturbation_operator, 0.5*(f0+f1), 0.5*(h0+h1))

                # Flatten for parallel solver
                H_flat = np.ascontiguousarray(H_eff.transpose(2, 3, 0, 1).reshape(-1, 6, 6))
                rho_flat = np.ascontiguousarray(rho_to_evolve.transpose(2, 3, 0, 1).reshape(-1, 6, 6))
                
                # Solve on worker processes
                rho_new_flat = parallel_density_evolution(H_flat, rho_flat, dt, executor)
                rho_new = rho_new_flat.reshape(Ny, Nx, 6, 6).transpose(2, 3, 0, 1)

                # Final relaxation
                if Gamma != 0:
                    rho_new = relax_rho(rho_new, rho_eq, dt/2, Gamma)
                
                t_evolve += time.perf_counter() - start

                # 3. Convergence Check
                err = np.max(np.abs(rho_new - rho_guess))
                rho_guess = rho_new

                if not do_freeze:
                    # Update fields for the next corrector iteration
                    f1 = helpers.H_fock(Kxmesh, Nk, rho_guess, a, V)
                    h1 = helpers.H_hartree(rho_guess, Nk, U, V)

                if err < tol:
                    break
            
            # Step complete
            rho = rho_guess

            # 4. Measurement
            start = time.perf_counter()
            rho_clean = np.ascontiguousarray(rho, dtype=np.complex128)
            rho_expvals[i] = measure(measure_operators, rho_clean) 
            t_measure += time.perf_counter() - start

    times = {'fock': t_fock, 'hartree': t_hartree, 'evolve': t_evolve, 'measure': t_measure}
    return dt * np.arange(N_points), rho_expvals, times

''' susceptibility obtained from temporal response, using Fourier transform. window exp(-eta*t) is applied '''
def susceptibility(time, signal, probe, eta, omega_cut, Nk):
    dt = time[1] - time[0]
    window = np.exp(- eta * time)
    
    signal_omega = np.fft.fft((signal - signal[0]) * window * dt) / Nk
    probe_omega = np.fft.fft(probe * window * dt)

    omega = 2*np.pi*np.fft.fftfreq(len(time), d=dt)

    pos = (omega > 0) * (omega < omega_cut)
    omega = omega[pos]
    signal_omega = signal_omega[pos]
    probe_omega = probe_omega[pos]

    return omega, signal_omega, probe_omega

''' optical conductivity calculated from susceptibility obtained from temporal response'''
def optical_conductivity(time, signal, probe, eta, omega_cut, Nk):
    omega, signal_omega, probe_omega = susceptibility(time, signal, probe, eta, omega_cut, Nk)
    sigma_omega = signal_omega / (-1j * omega * probe_omega)

    return omega, sigma_omega.real

def integral_omega(integrand, omega):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(integrand.real, omega)
    else:
        return np.trapz(integrand.real, omega)

def to_scalar_if_single(x):
    x = np.asarray(x)
    if x.size == 1:
        return float(x.item())
    return x