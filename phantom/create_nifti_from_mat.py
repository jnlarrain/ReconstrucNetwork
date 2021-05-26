from scipy import io
import ants


salida = io.loadmat('new_image.mat')['new_image']
entrada = io.loadmat('new_phase.mat')['new_phase']


# new_image = ants.make_image(entrada.shape)
new_image = ants.from_numpy(entrada)
ants.image_write(new_image, 'pha.nii.gz')


# new_image = ants.make_image(salida.shape)
new_image = ants.from_numpy(salida)
ants.image_write(new_image, 'sus.nii.gz')





