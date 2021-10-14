"""
Script para la generacion de un solo dato
"""
import numpy as np
from numpy.random import randint, choice
from data_generator.figures.K_space import calculate_k
from data_generator.figures.spherestf import sphere
from data_generator.figures.cylinderstf import cylinder
from random import randint, choice


class Data:
    """
    Clase para generar una imagen

    """
    def __init__(self, size, susceptibilidad_interna=1, susceptibildad_externa=9, bias_susceptibilidad=0,
                 numero_esferas=randint(1, 1024), numero_cilindros=randint(1, 512)):
        """
        Constructor de la clase, define los parametros generales de la imagen

        Parameters
        ----------
        size: int
            tamaño de una arista del cubo:
        susceptibilidad_interna: float
            rango de susceptibilidad interna en valor absoluto.
        susceptibildad_externa: float
            rango de susceptibilidad externa en valor absoluto.
        bias_susceptibilidad: float
            bias aplicado al rango de las susceptibilidades.
        numero_esferas: int
            numero de esferas a crear, en caso de no ser otrogado es random.
        numero_cilindros: int
            numero de cilindros a crear, en caso de no ser otorgado es random.
        """
        self.size = size
        self.field_of_view = [size, ]*3
        self.numero_esferas = numero_esferas
        self.numero_cilindros = numero_cilindros
        # self.range_center = [size // 4, size // 4 * 3]
        self.range_center = [0, size - 1]
        self.range_radio = [4, size // 4]
        self.c_range = 8  # rango en que los radios de los cilindros son mas pequeños
        self.sus_in = susceptibilidad_interna
        self.sus_ex = susceptibildad_externa
        self.bias = bias_susceptibilidad

    def kspace(self, original_fov, radio, range_center, chi, center_choice=(), _cylinder=False):
        """
        Funcion para generar k space de una esfera o cilidro de forma random
        Parameters
        ----------
        original_fov: iterable

        radio: iterblale
            rango valor entero del radio de las figuras TODO: agregar valores fraccionales para simulaciones mas completas
        range_center: iterable
            rango de valores para generar el centro de las figuras, en formato [min, max]
        chi: float
            valor de la susceptibilidad
        center_choice:
        center_choice: bool
            flag para indicar que el rango de centros ya fue seleccionado por lo que solo se debe tomar uno al azar.
        _cylinder: bool
            flag paa indicar si se esta creando un cilindro o no
        Returns
        -------

        """
        suscep = np.random.uniform(0, chi) * np.random.choice([-1, 1]) + self.bias
        radio = randint(*radio)
        if _cylinder:
            p1 = np.array([randint(*range_center), randint(*range_center), randint(*range_center)], dtype='float32')
            p2 = np.array([randint(*range_center), randint(*range_center), randint(*range_center)], dtype='float32')
            k = (*calculate_k(original_fov, original_fov, points=p1),)
            return k, radio, suscep, p1, p2
        if center_choice:
            centers = [choice(range_center), choice(range_center), choice(range_center)]
        else:
            centers = [randint(*range_center), randint(*range_center), randint(*range_center)]

        k = (*calculate_k(original_fov, original_fov, center=centers),)
        return k, radio, suscep

    @property
    def new_image(self):
        """
        Funcion para generar una imagen sin ruido ni campo externo

        Parameters
        ----------
        Usa los parametros dados en el constructor

        Returns
        -------
        tuple con los arrays de la susceptibilidad, fase y magnitud
        """
        susceptibilidad = np.zeros(self.field_of_view)
        phase = np.zeros(self.field_of_view)
        magnitud = np.zeros(self.field_of_view)
        for _ in range(self.numero_esferas):
            k, radio, suscep = self.kspace(self.field_of_view, self.range_radio, self.range_center, self.sus_in)
            _suscep, campo, _ = sphere(k, radio, suscep, self.sus_ex+self.bias)
            susceptibilidad += _suscep.numpy()
            magnitud += np.where(_suscep.numpy() == suscep, 1., 0.)
            phase += campo.numpy()
        for _ in range(self.numero_cilindros):
            k, radio, suscep, p1, p2 = self.kspace(self.field_of_view, np.array(self.range_radio) // self.c_range,
                                                   self.range_center, self.sus_in, _cylinder=True)
            _suscep, campo = cylinder(k, self.field_of_view, p1, p2, radio, suscep, self.sus_ex+self.bias)
            susceptibilidad += _suscep.numpy()
            magnitud += np.where(_suscep.numpy() == suscep, 1., 0.)
            phase += campo.numpy()
        return susceptibilidad, phase, np.where(magnitud > 0, 1., 0.)

    def foco_externo(self, number, fov, _susceptibilidad, _radio=False):
        """
        funcion para agregar campos externos a una imagen
        Parameters
        ----------
        number: int
            numero de focos externos a agregar
        fov: iterable
            tamaño del campo de vision externo donde se van a agregar los focos
        Returns
        -------

        """
        if not _radio:
            _radio = [10, 20]
        background = np.zeros(self.field_of_view)
        posible_range = [x for x in range(fov[0]) if x < int(fov[0] // 8) or x > int(fov[0] // 8 * 7)]
        for _ in range(number):
            k, radio, suscep = self.kspace(fov, _radio, posible_range, _susceptibilidad, True)
            suscep = _susceptibilidad
            susceptibilidad, campo, _ = sphere(k, radio, suscep, self.sus_ex+self.bias)
            cnt = self.field_of_view[0]
            background += campo[cnt // 2:cnt // 2 + cnt, cnt // 2:cnt // 2 + cnt, cnt // 2:cnt // 2 + cnt].numpy()
        return background

