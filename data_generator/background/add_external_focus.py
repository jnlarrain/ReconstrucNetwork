import numpy as np
from data_generator.figures.spherestf import sphere


def out_spheres(number, original_fov, bigger_fov, size):
    # campo inicial
    background = np.zeros(original_fov)
    # rango posible para los centros
    posible_range = [x for x in range(bigger_fov[0]) if x < size // 8 or x > int(size * 14 / 8)]
    for _ in range(number):
        k, radio, suscep = randoms(bigger_fov, [1, size // 16 - 1], posible_range, 10, True)
        susceptibilidad, campo, _ = sphere(k, radio, suscep, 0)
        # susceptibilidad /= 1e-6
        cnt = original_fov[0]
        background += susceptibilidad[cnt // 2:cnt // 2 + cnt, cnt // 2:cnt // 2 + cnt, cnt // 2:cnt // 2 + cnt]
    return background