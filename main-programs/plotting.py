import matplotlib.pyplot as plt
import numpy as np
import matplotlib
plt.rcParams.update({'savefig.dpi':300, 'axes.labelweight':'normal'})
matplotlib.rcParams['axes.linewidth'] = 0.8
from matplotlib import rc
preamble = r'''
\usepackage{physics} \usepackage{upgreek} \usepackage{mhchem} \usepackage{bm} \usepackage{amsfonts} \usepackage{amssymb} \usepackage{dsfont}
'''
plt.rc('text.latex', preamble=preamble)
rc('text', usetex=True)

def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(matplotlib.colors.to_rgb(c1))
    c2=np.array(matplotlib.colors.to_rgb(c2))
    return matplotlib.colors.to_hex((1-mix)*c1 + mix*c2)

def bands(energije, barve, mu, name='bands', a=3.51, b=15.79, col1='firebrick', col2='blue'):
    fig, ax = plt.subplots(figsize=(6,4), facecolor='white')
    #mu = 0.5 * (np.min(energije[2]) + np.max(energije[1]))
    Ny, Nx = energije.shape[-2:]
    kX = 2*np.pi / a
    kY = 2*np.pi / b

    # Gamma - X
    for i in range(6):
        K = (Nx//2-1)*kX + (Ny//2-1)*kY + np.arange(Nx//2)*kX
        E = energije[i, Ny//2, Nx//2:]
        for g in range(len(K)-1):
            shade = np.mean(barve[i, Ny//2, Nx//2:][g:g+2])
            if shade > 1.0:
                shade = 1.0
            plt.plot(K[g:g+2], E[g:g+2]- mu, color=colorFader(col1, col2, shade))

    # Z - Gamma
    E_c, E_v = np.max(energije[:Ny//2, Nx//2]), np.min(energije[:Ny//2, Nx//2])
    gap = E_c - E_v
    for i in range(6):
        K = (Nx//2-1)*kX + np.arange(Ny//2)*kY
        E = energije[i, :Ny//2, Nx//2]
        for g in range(len(K)-1):
            shade = np.mean(barve[i, :Ny//2, Nx//2][g:g+2])
            if shade > 1.0:
                shade = 1.0
            plt.plot(K[g:g+2], E[g:g+2]- mu, color=colorFader(col1, col2, shade))


    # Z - M
    for i in range(6):
        K = np.arange(Nx//2)*kX
        E = energije[i, 0, Nx//2:][::-1]
        for g in range(len(K)-1):
            shade = np.mean(barve[i, 0, Nx//2:][::-1][g:g+2])
            print(shade)
            if shade > 1.0:
                shade = 1.0
            plt.plot(K[g:g+2], E[g:g+2] - mu, color=colorFader(col1, col2, shade))

    ax.set_xticks([0, (Nx//2 -1)*kX, (Nx//2 -1)*kX + (Ny//2 - 1)*kY, (Nx//2 -1)*kX + (Ny//2 - 1)*kY + (Nx//2-1)*kX],)
    ax.set_xticklabels(['$M$', '$Z$', r'$\Gamma$', '$X$'])

    plt.xticks(fontsize=15), plt.yticks(fontsize=18)
    plt.ylabel(r'$\varepsilon_{\bm{k}} - E_F\,[\text{eV}]$', fontsize=18)

    plt.title(r'band structure along $M$--$Z$--$\Gamma$--$X$', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{name}.pdf')
    
def optics(omega0, results_x, results_y, results_xy, results_yx, name='optics'):
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

    plt.tight_layout()
    plt.savefig(f'{name}.pdf')