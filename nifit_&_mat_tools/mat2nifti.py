import scipy.io
import ants
import os

import numpy as np

path = "."
for file in os.listdir(path):

    if ".mat" in file:
        print(file)
        mat = scipy.io.loadmat(f"{os.getcwd()}\\{file}")
        try:
            ants.image_write(
                ants.from_numpy(mat["mask"]), f"invivo_mask_{file[:-4]}.nii"
            )
            ants.image_write(
                ants.from_numpy(mat["mask2"]), f"invivo_mask2_{file[:-4]}.nii"
            )
        except:
            try:
                ants.image_write(
                    ants.from_numpy(np.cast[np.float32](mat["mag_use"])),
                    f"invivo_mask_{file[:-4]}.nii",
                )
            except:
                pass
        try:

            ants.image_write(
                ants.from_numpy(mat["new_phase1"]), f"invivo_phase_{file[:-4]}.nii"
            )
            ants.image_write(
                ants.from_numpy(mat["new_phase2"]), f"invivo_phase2_{file[:-4]}.nii"
            )
        except:

            try:
                ants.image_write(
                    ants.from_numpy(mat["phase_use"]),
                    f"invivo_phase_{file[:-4]}.nii",
                )
            except:
                pass
