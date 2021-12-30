import scipy.io
import ants
import os

path = 'C:\\Users\\jlarr\\Desktop\\challenge\\'
for file in os.listdir(path):
    if '.mat' in file:
        try:
            mat = scipy.io.loadmat(f'C:\\Users\\jlarr\\Desktop\\challenge\\{file}')
            ants.image_write(ants.from_numpy(mat['mask']),f'invivo_mask_{file}.nii')
            ants.image_write(ants.from_numpy(mat['new']),f'invivo_phase_iso_{file}.nii')
            print(mat.keys())
        except:
            continue
