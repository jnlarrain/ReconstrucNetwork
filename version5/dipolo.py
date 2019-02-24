import numpy as np


FOV = 64

dipole = [[[(1/3 - (z**2/(x**2 + y**2 + z**2))) for z in range(1,FOV + 1)]for y in range(1,FOV + 1)] for x in range(1,FOV + 1)]
dipole = np.asarray(dipole)
dipole = abs(np.fft.ifftn(dipole))


# sacar la transformada de fourier analitica





