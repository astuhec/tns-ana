import matplotlib.pyplot as plt
import numpy as np

def optics(omega0, results_x, results_y, results_xy, results_yx):
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

    plt.show()