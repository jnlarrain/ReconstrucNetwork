
import os
import ants
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np

size = 96

# tfrecords path
path = 'F:/' + str(size) + 'data/train/'

datos_train = [os.path.join(path, x) for x in os.listdir(path) if '.nii' in x]
mascaras = [ants.image_read(x).numpy() for x in datos_train if 'mag_final' in x]
input_phase = [ants.image_read(x).numpy() for x in datos_train if 'phase.nii' in x]
output_phase = [ants.image_read(x).numpy() for x in datos_train if 'phase_final' in x]


results = []

for i in range(len(mascaras)):
    diff = output_phase[i]*(1-mascaras[i]) - input_phase[i]*(1-mascaras[i])
    std = diff.std()
    print(std/np.max(np.abs(input_phase)))




