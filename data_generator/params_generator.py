import numpy as np
from numpy.random import randint, choice
from data_generator.figures.K_space import calculate_k


def create_random_figures(original_fov, radio, range_center, chi, center_choice=(), _cylinder=False, bigger_fov=()):
    """
    Funcion para generar k space de una esfera o cilidro de forma random
    Parameters
    ----------
    original_fov: iterable

    radio:
    range_center: iterable
    chi: float
        valor de la susceptibilidad
    center_choice:
    center_choice bool
        flag para indicar que el rango de centros ya fue seleccionado por lo que solo se debe tomar uno al azar.
    _cylinder
    bigger_fov:

    Returns
    -------

    """
    suscep = np.random.normal(0, chi) * 1e-6 * np.random.choice([-1, 1])
    radio = randint(*radio)
    if _cylinder:
        p1 = np.array([randint(*range_center), randint(*range_center), randint(*range_center)], dtype='float32')
        p2 = np.array([randint(*range_center), randint(*range_center), randint(*range_center)], dtype='float32')
        k = (*calculate_k(original_fov, bigger_fov, points=p1),)
        return k, radio, suscep, p1, p2
    if center_choice:
        centers = [choice(range_center), choice(range_center), choice(range_center)]
    else:
        centers = [randint(*range_center), randint(*range_center), randint(*range_center)]

    k = (*calculate_k(original_fov, original_fov, center=centers),)
    return k, radio, suscep
