import numpy as np
from data_generator.figures.K_space import calculate_k
from data_generator.figures.spherestf import sphere
import SimpleITK as sitk
import matplotlib.pyplot as plt

#######################################################################################################################

fov = [96, ]*3
number_figures = 1
c = fov[0]//2
rango = np.linspace(c-13, c+13)
cho = np.random.choice
centers = [[c, ]*3, cho(rango, 3), cho(rango, 3), cho(rango, 3), cho(rango, 3)]

#######################################################################################################################


def create_testing_figure(fov, centers, inner_sus, out_sus):
    radio = 12
    k = (*calculate_k(fov, fov, center=centers),)
    x, y, _ = sphere(k, radio, inner_sus, out_sus)
    return x, y


sus = 1e-6
sus_ext = 0

for iteration in range(20):
    sus_img = np.zeros(fov)
    pha_img = np.zeros(fov)
    for i in range(len(centers)):
        chi, pha = create_testing_figure(fov, centers[i], sus, sus_ext)
        sus_img += chi
        pha_img += pha
    sus -= 5e-7
    plt.hist(np.ravel(np.log(pha_img+1e-10)), bins=50)
    plt.title('sus: interna: {:.2e}   externa: {:.2e}'.format(sus, sus_ext))
    plt.show()
    sitk.WriteImage(sitk.GetImageFromArray(sus_img),
                    'F:/tesis/output_analisis/sus_sus{:.2e}ext{:.2e}una_esfera.nii.gz'.format(sus, sus_ext))
    sitk.WriteImage(sitk.GetImageFromArray(pha_img),
                    'F:/tesis/output_analisis/pha_sus{:.2e}ext{:.2e}una_esfera.nii.gz'.format(sus, sus_ext))























