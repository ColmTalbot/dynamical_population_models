import unittest

import numpy as np

from bilby.core.prior import Uniform, PriorDict

from gwpopulation.cupy_utils import xp
from gwpopulation.models import mass


class TestPrimaryMassRatio(unittest.TestCase):

    def setUp(self):
        self.m1s = np.linspace(3, 100, 1000)
        self.qs = np.linspace(0.01, 1, 500)
        m1s_grid, qs_grid = xp.meshgrid(self.m1s, self.qs)
        self.dataset = dict(mass_1=m1s_grid, mass_ratio=qs_grid)
        self.power_prior = PriorDict()
        self.power_prior['alpha'] = Uniform(minimum=-4, maximum=12)
        self.power_prior['beta'] = Uniform(minimum=-4, maximum=12)
        self.power_prior['mmin'] = Uniform(minimum=3, maximum=10)
        self.power_prior['mmax'] = Uniform(minimum=40, maximum=100)
        self.gauss_prior = PriorDict()
        self.gauss_prior['lam'] = Uniform(minimum=0, maximum=1)
        self.gauss_prior['mpp'] = Uniform(minimum=20, maximum=60)
        self.gauss_prior['sigpp'] = Uniform(minimum=0, maximum=10)
        self.n_test = 10

    def test_dynamic(self):
        parameters = self.power_prior.sample()
        parameters.update(self.gauss_prior.sample())
        parameters = dict(
            alpha=2.0,
            mmin=5.0,
            mmax=45.0,
            lam=0.1,
            mpp=35.0,
            sigpp=1.0,
            beta=1.0,
            branch_1=0.12,
            branch_2=0.01
        )
        prob = mass.two_component_primary_mass_ratio_dynamical(
            dataset=self.dataset, **parameters)
        self.assertTrue(all(
            prob[self.dataset["mass_1"] * self.dataset["mass_ratio"] <=
                 parameters["mmin"]] == 0))
