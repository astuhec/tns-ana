import matplotlib.pyplot as plt
import numpy as np
import matplotlib

def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(matplotlib.colors.to_rgb(c1))
    c2=np.array(matplotlib.colors.to_rgb(c2))
    return matplotlib.colors.to_hex((1-mix)*c1 + mix*c2)

def bands(energije, barve, mu, name='bands', a=3.51, b=15.79, col1='firebrick', col2='blue'):
    fig, ax = plt.subplots(figsize=(8,4), facecolor='white')
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
            plt.plot(K[g:g+2], E[g:g+2]- mu, color=colorFader(col1, col2, shade))

    # Z - Gamma
    E_c, E_v = np.max(energije[:Ny//2, Nx//2]), np.min(energije[:Ny//2, Nx//2])
    gap = E_c - E_v
    for i in range(6):
        K = (Nx//2-1)*kX + np.arange(Ny//2)*kY
        E = energije[i, :Ny//2, Nx//2]
        for g in range(len(K)-1):
            shade = np.mean(barve[i, :Ny//2, Nx//2][g:g+2])
            plt.plot(K[g:g+2], E[g:g+2]- mu, color=colorFader(col1, col2, shade))


    # Z - M
    for i in range(6):
        K = np.arange(Nx//2)*kX
        E = energije[i, 0, Nx//2:][::-1]
        for g in range(len(K)-1):
            shade = np.mean(barve[i, 0, Nx//2:][::-1][g:g+2])
            plt.plot(K[g:g+2], E[g:g+2] - mu, color=colorFader(col1, col2, shade))

    ax.set_xticks([0, (Nx//2 -1)*kX, (Nx//2 -1)*kX + (Ny//2 - 1)*kY, (Nx//2 -1)*kX + (Ny//2 - 1)*kY + (Nx//2-1)*kX],)
    ax.set_xticklabels(['$M$', '$Z$', r'$\Gamma$', '$X$'])

    plt.xticks(fontsize=15), plt.yticks(fontsize=18)
    plt.ylabel(r'$\varepsilon_{\bm{k}} - E_F\,[\text{eV}]$', fontsize=18)

    plt.title(r'band structure along $M$--$Z$--$\Gamma$--$X$', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{name}.pdf')