import numpy as np
import os
import nibabel as nib
import pytube


if __name__ == '__main__':
    path = 'F:/SNR1/Sim1Snr1/Phase.nii.gz'
    img = nib.load(path)
    print(img.shape)
    data = img.get_fdata()
    print(data.shape)
    print(img.header)




