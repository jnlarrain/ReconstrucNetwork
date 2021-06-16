"""

funciones de Romeo


usage: <PROGRAM> [-m MAGNITUDE] [-o OUTPUT] [-t ECHO-TIMES] [-k MASK]
                 [-e UNWRAP-ECHOES] [-w WEIGHTS] [-B]
                 [--phase-offset-correction] [-i]
                 [--template TEMPLATE] [-N] [--no-rescale]
                 [--threshold THRESHOLD] [-v] [-g] [-q] [-Q]
                 [-s MAX-SEEDS] [--merge-regions] [--correct-regions]
                 [--wrap-addition WRAP-ADDITION]
                 [--temporal-uncertain-unwrapping] [--version] [-h]
                 [phase]

positional arguments:
  phase                 The phase image used for unwrapping

optional arguments:
  -m, --magnitude MAGNITUDE
                        The magnitude image (better unwrapping if
                        specified)
  -o, --output OUTPUT   The output path or filename (default:
                        "unwrapped.nii")
  -t, --echo-times ECHO-TIMES
                        The relative echo times required for temporal
                        unwrapping specified in array or range syntax
                        (eg. "[1.5,3.0]" or "3.5:3.5:14"). (default is
                        ones(<nr_of_time_points>) for multiple volumes
                        with the same time) Warning: No spaces
                        allowed!! ("[1, 2, 3]" is invalid!)
  -k, --mask MASK       nomask | robustmask | <mask_file> (default:
                        "robustmask")
  -e, --unwrap-echoes UNWRAP-ECHOES
                        Load only the specified echoes from disk
                        (default: ":")
  -w, --weights WEIGHTS
                        romeo | romeo2 | romeo3 | romeo4 | bestpath |
                        <4d-weights-file> | <flags>. <flags> are four
                        bits to activate individual weights (eg.
                        "1010"). The weights are (1)phasecoherence
                        (2)phasegradientcoherence (3)phaselinearity
                        (4)magcoherence (default: "romeo")
  -B, --compute-B0      Calculate combined B0 map in [Hz]. Phase
                        offset                correction might be
                        necessary if not coil-combined with
                        MCPC3Ds/ASPIRE.
  --phase-offset-correction
                        Applies the MCPC3Ds method to perform phase
                        offset determination and removal (for
                        multi-echo).
  -i, --individual-unwrapping
                        Unwraps the echoes individually (not
                        temporal). This might be necessary if there is
                        large movement (timeseries) or
                        phase-offset-correction is not applicable.
  --template TEMPLATE   Template echo that is spatially unwrapped and
                        used for temporal unwrapping (type: Int64,
                        default: 2)
  -N, --no-mmap         Deactivate memory mapping. Memory mapping
                        might cause problems on network storage
  --no-rescale          Deactivate rescaling of input images. By
                        default the input phase is rescaled to the
                        range [-π;π]. This option allows inputting
                        already unwrapped phase images without
                        wrapping them first.
  --threshold THRESHOLD
                        <maximum number of wraps>. Threshold the
                        unwrapped phase to the maximum number of wraps
                        and sets exceeding values to 0 (type: Float64,
                        default: Inf)
  -v, --verbose         verbose output messages
  -g, --correct-global  Phase is corrected to remove global n2π phase
                        offset. The median of phase values (inside
                        mask if given) is used to calculate the
                        correction term
  -q, --write-quality   Writes out the ROMEO quality map as a 3D image
                        with one value per voxel
  -Q, --write-quality-all
                        Writes out an individual quality map for each
                        of the ROMEO weights.
  -s, --max-seeds MAX-SEEDS
                        EXPERIMENTAL! Sets the maximum number of seeds
                        for unwrapping. Higher values allow more
                        seperated regions. (type: Int64, default: 1)
  --merge-regions       EXPERIMENTAL! Spatially merges neighboring
                        regions after unwrapping.
  --correct-regions     EXPERIMENTAL! Performed after merging. Brings
                        the median of each region closest to 0 (mod
                        2π).
  --wrap-addition WRAP-ADDITION
                        [0;π] EXPERIMENTAL! Usually the true phase
                        difference of neighboring voxels cannot exceed
                        π to be able to unwrap them. This setting
                        increases the limit and uses 'linear
                        unwrapping' of 3 voxels in a line. Neighbors
                        can have (π + wrap-addition) phase difference.
                        (type: Float64, default: 0.0)
  --temporal-uncertain-unwrapping
                        EXPERIMENTAL! Uses spatial unwrapping on
                        voxels that have high uncertainty values after
                        temporal unwrapping.
  --version             show version information and exit
  -h, --help            show this help message and exit
"""

import os

img_num = 4
read_path = 'F:\\tesis\\invivo_uc\\img{}'.format(img_num)
out_path = 'F:\\tesis\\invivo_uc\\rimg{}'.format(img_num)
phase_list_path = list(filter(lambda x: 'ph.nii' in x, os.listdir(read_path)))
mag_list_path = list(filter(lambda x: '.nii' in x and 'ph.nii' not in x and 'imaginary.nii' not in x and
                                      'real.nii' not in x, os.listdir(read_path)))

for element in range(len(phase_list_path)):
    # os.system('romeo {} -m {} -o {}'.format(os.path.join(read_path, phase_list_path[element]),
    #                                         os.path.join(read_path, mag_list_path[element]),
    #                                         os.path.join(out_path, phase_list_path[element][:-7])))
    os.system('romeo {} -o {}'.format(os.path.join(read_path, phase_list_path[element]),
                                      os.path.join(out_path, phase_list_path[element][:-7])))
