"""
Script para agregar ruido complejo a una imagen
"""

import numpy as np
from random import uniform


def add_complex_noise(magnitud, phase, norm=False, SNR=384): #256 mucho ruido 1024 muy poco
    """
    funcion para agregar el ruido en radianes

    Parameters
    ----------
    magnitud: numpy array
        magnitud de la imagen a agregar ruido
    phase: numpy array
        fase de la imagen a agregar ruido
    norm: Bool
        flag para indicar que los datos necesitan conversion radianes o no
    SNR: int
        razon señal a ruido
    Returns
    -------


    """
    # ver si agregar B0
    phase = np.cast['float64'](phase)
    gamma = 267.522e6  # constante giromagnetica en radianes
    TE = uniform(10e-3, 40e-3)
    phase *= gamma * TE
    constante_normalizacion_minimo = np.min(phase)
    phase -= constante_normalizacion_minimo
    constante_normalizacion_maximo = np.max(phase)
    phase /= constante_normalizacion_maximo
    phase *= 2*np.pi
    phase -= np.pi
    complejo = magnitud*np.exp(1j*phase)    # ver como no enmascarar la phase con la magnitud
    std_ruido = np.max(magnitud)/SNR
    ruido = np.random.normal(0, std_ruido, phase.shape) + np.random.normal(0, std_ruido, phase.shape)*1j
    complejo += ruido
    phase = np.angle(complejo)
    phase += np.pi
    phase /= 2*np.pi
    phase *= constante_normalizacion_maximo
    phase += constante_normalizacion_minimo
    phase /= (gamma * TE)
    phase = np.cast['float32'](phase)*magnitud
    return np.abs(complejo), phase