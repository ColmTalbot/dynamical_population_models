import os

import numpy as np

from gwpopulation.conversions import mu_chi_var_chi_max_to_alpha_beta_max
from gwpopulation.utils import beta_dist, powerlaw
from gwpopulation.cupy_utils import trapz, xp
from gwpopulation.models.mass import (
    two_component_primary_mass_ratio,
    two_component_single,
)
from gwpopulation.models.spin import iid_spin_magnitude_beta

NAN_INF = xp.nan_to_num(xp.inf)


def two_component_primary_mass_ratio_dynamical_with_spins(
    dataset,
    alpha,
    beta,
    mmin,
    mmax,
    lam,
    mpp,
    sigpp,
    alpha_chi,
    beta_chi,
    delta_chi,
    branch_1=0.12,
    branch_2=0.01,
):
    """
    Power law model for two-dimensional mass distribution, modelling primary
    mass and conditional mass ratio distribution.

    p(m1, q) = p(m1) * p(q | m1)

    We also include the effect of dynamical mergers leading to 1.5 and 2nd
    generation mergers.

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays for 'mass_1' and 'mass_ratio'.
    alpha: float
        Negative power law exponent for more massive black hole.
    mmin: float
        Minimum black hole mass.
    mmax: float
        Maximum black hole mass.
    beta: float
        Power law exponent of the mass ratio distribution.
    lam: float
        Fraction of black holes in the Gaussian component.
    mpp: float
        Mean of the Gaussian component.
    sigpp: float
        Standard deviation fo the Gaussian component.
    delta_chi: float
        Fraction of black holes in the low spin component
    branch_1: float
        Ratio of 1.5 generation mergers to 1st generation mergers.
        The default value comes from a conversation with Eric Thrane.
    branch_2: float
        ratio of 2nd generation mergers to 1st generation mergers.
        The default value comes from a conversation with Eric Thrane.
    """
    btot = 1 + branch_1 + branch_2

    fraction_1_5 = branch_1 / btot
    fraction_2 = branch_2 / btot
    fraction_1 = 1 - fraction_1_5 - fraction_2
    assert fraction_1 >= 0, "Branching fractions greater than 1."

    first_generation = first_generation_mass_spin(
        dataset=dataset,
        alpha=alpha,
        beta=beta,
        mmin=mmin,
        mmax=mmax,
        lam=lam,
        mpp=mpp,
        sigpp=sigpp,
        alpha_chi=alpha_chi,
        beta_chi=beta_chi,
        delta_chi=delta_chi,
    )

    one_point_five_generation = one_point_five_generation_mass_spin(
        dataset=dataset,
        alpha=alpha,
        beta=beta,
        mmin=mmin,
        mmax=mmax,
        lam=lam,
        mpp=mpp,
        sigpp=sigpp,
        alpha_chi=alpha_chi,
        beta_chi=beta_chi,
        delta_chi=delta_chi,
    )

    second_generation = second_generation_mass_spin(
        dataset=dataset,
        alpha=alpha,
        beta=beta,
        mmin=mmin,
        mmax=mmax,
        lam=lam,
        mpp=mpp,
        sigpp=sigpp,
    )

    return (
        fraction_1 * first_generation
        + fraction_1_5 * one_point_five_generation
        + fraction_2 * second_generation
    )


def first_generation_mass_spin(
    dataset,
    alpha,
    beta,
    mmin,
    mmax,
    lam,
    mpp,
    sigpp,
    alpha_chi,
    beta_chi,
    delta_chi,
):
    first_generation_mass = two_component_primary_mass_ratio(
        dataset=dataset,
        alpha=alpha,
        beta=beta,
        mmin=mmin,
        mmax=mmax,
        lam=lam,
        mpp=mpp,
        sigpp=sigpp,
    )
    first_generation_spin = first_generation_spin_magnitude(
        dataset["a_1"],
        alpha=alpha_chi,
        beta=beta_chi,
        delta=delta_chi,
        a_max=1,
    ) * first_generation_spin_magnitude(
        dataset["a_2"],
        alpha=alpha_chi,
        beta=beta_chi,
        delta=delta_chi,
        a_max=1,
    )
    return first_generation_mass * first_generation_spin


def one_point_five_generation_mass_spin(
    dataset,
    alpha,
    beta,
    mmin,
    mmax,
    lam,
    mpp,
    sigpp,
    alpha_chi,
    beta_chi,
    delta_chi,
):
    params = dict(
        mmin=mmin * 2, mmax=mmax * 2, lam=lam, mpp=mpp * 2, sigpp=sigpp * 2
    )

    one_point_five_generation_mass = two_component_single(
        dataset["mass_1"], alpha=alpha, **params
    ) * one_point_five_generation_mass_ratio(
        dataset, spectal_index=beta * 1.5, mmin=mmin
    )

    alpha_2g, beta_2g, _ = mu_chi_var_chi_max_to_alpha_beta_max(
        mu_chi=0.67, var_chi=0.01, amax=1
    )

    one_point_five_generation_spin = beta_dist(
        dataset["a_1"], scale=1, alpha=alpha_2g, beta=beta_2g
    ) * first_generation_spin_magnitude(
        dataset["a_2"],
        alpha=alpha_chi,
        beta=beta_chi,
        delta=delta_chi,
        a_max=1,
    )

    return one_point_five_generation_mass * one_point_five_generation_spin


def second_generation_mass_spin(
    dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp,
):
    second_generation_mass = two_component_primary_mass_ratio(
        dataset=dataset,
        alpha=alpha,
        beta=beta * 4,
        mmin=mmin * 2,
        mmax=mmax * 2,
        lam=lam,
        mpp=mpp * 2,
        sigpp=sigpp * 2,
    )

    alpha_2g, beta_2g, _ = mu_chi_var_chi_max_to_alpha_beta_max(
        mu_chi=0.67, var_chi=0.01, amax=1
    )

    second_generation_spin = iid_spin_magnitude_beta(
        dataset=dataset, alpha_chi=alpha_2g, beta_chi=beta_2g, amax=1
    )

    return second_generation_mass * second_generation_spin

class EmpiricalBranchingFraction(object):
    def __init__(self):
        self.retention_file = self.__retention_file__()
        branching_dataset = np.load(
            self.retention_file,
            allow_pickle=True,
            encoding="latin1",
        )
        self.a_1_array = xp.asarray(branching_dataset["a1"])
        self.a_2_array = xp.asarray(branching_dataset["a2"])
        self.mass_ratio_array = xp.asarray(branching_dataset["q"])
        self.retention_fraction = xp.asarray(
            branching_dataset["interpolated_retention_fraction"])
        self.mass_1s = xp.linspace(2, 200, 2000)
        self.mass_ratio_grid, self.mass_1_grid = xp.meshgrid(
            self.mass_ratio_array, self.mass_1s
        )
        self.first_generation_data = dict(
            mass_1=self.mass_1_grid, mass_ratio=self.mass_ratio_grid
        )
    def __retention_file__(self):
        return os.path.join(os.path.dirname(__file__), "grid_dict_5e5")
    def __call__(
        self,
        dataset,
        alpha,
        beta,
        mmin,
        mmax,
        lam,
        mpp,
        sigpp,
        alpha_chi,
        beta_chi,
        delta_chi,
    ):
        branching_ratio = self.compute_branching_ratio(
            alpha=alpha,
            beta=beta,
            mmin=mmin,
            mmax=mmax,
            lam=lam,
            mpp=mpp,
            sigpp=sigpp,
            alpha_chi=alpha_chi,
            beta_chi=beta_chi,
            delta_chi=delta_chi,
        )
        return two_component_primary_mass_ratio_dynamical_with_spins(
            dataset=dataset,
            alpha=alpha,
            beta=beta,
            mmin=mmin,
            mmax=mmax,
            lam=lam,
            mpp=mpp,
            sigpp=sigpp,
            alpha_chi=alpha_chi,
            beta_chi=beta_chi,
            delta_chi=delta_chi,
            branch_1= 0.5 * branching_ratio,
            branch_2= 0.125 * branching_ratio ** 2 
        )

    def compute_branching_ratio(
        self,
        alpha,
        beta,
        mmin,
        mmax,
        lam,
        mpp,
        sigpp,
        alpha_chi,
        beta_chi,
        delta_chi,
        a_max=1,
    ):

        probability = xp.einsum(
            "i,j,k->ijk",
            self.first_generation_mass_ratio(
                alpha=alpha,
                beta=beta,
                mmin=mmin,
                mmax=mmax,
                lam=lam,
                mpp=mpp,
                sigpp=sigpp,
            ),
            first_generation_spin_magnitude_grid(
                self.a_1_array,
                alpha=alpha_chi,
                beta=beta_chi,
                delta=delta_chi,
                a_max=a_max,
            ),
            first_generation_spin_magnitude_grid(
                self.a_2_array,
                alpha=alpha_chi,
                beta=beta_chi,
                delta=delta_chi,
                a_max=a_max,
            ),
        )
        branching_ratio = trapz(
            trapz(
                trapz(
                    probability * self.retention_fraction,
                    self.mass_ratio_array,
                ),
                self.a_2_array,
            ),
            self.a_1_array,
        )
        branching_ratio = min(branching_ratio, 1)
        return branching_ratio

    def first_generation_mass_ratio(
        self, alpha, beta, mmin, mmax, lam, mpp, sigpp
    ):
        first_generation_mass = two_component_primary_mass_ratio(
            dataset=self.first_generation_data,
            alpha=alpha,
            beta=beta,
            mmin=mmin,
            mmax=mmax,
            lam=lam,
            mpp=mpp,
            sigpp=sigpp,
        )
        return trapz(first_generation_mass, self.mass_1s, axis=0)


class EmpiricalBranchingFractionNoSpin(EmpiricalBranchingFraction):
    def __call__(
        self,
        dataset,
        alpha,
        beta,
        mmin,
        mmax,
        lam,
        mpp,
        sigpp,
        alpha_chi,
        beta_chi,
        delta_chi,
    ):
        branching_ratio = self.compute_branching_ratio(
            alpha=alpha,
            beta=beta,
            mmin=mmin,
            mmax=mmax,
            lam=lam,
            mpp=mpp,
            sigpp=sigpp,
            alpha_chi=alpha_chi,
            beta_chi=beta_chi,
            delta_chi=delta_chi,
        )
        return two_component_primary_mass_ratio_dynamical_without_spins(
            dataset=dataset,
            alpha=alpha,
            beta=beta,
            mmin=mmin,
            mmax=mmax,
            lam=lam,
            mpp=mpp,
            sigpp=sigpp,
            branch_1= 0.5 * branching_ratio,
            branch_2= 0.125 * branching_ratio ** 2
        )

class EmpiricalBranchingFraction_1e7(EmpiricalBranchingFraction):
    def __retention_file__(self):
        return os.path.join(os.path.dirname(__file__), "grid_dict_1e7")

class EmpiricalBranchingFraction_1e8(EmpiricalBranchingFraction):
    def __retention_file__(self):
        return os.path.join(os.path.dirname(__file__), "grid_dict_1e8")

class EmpiricalBranchingFractionNoSpin_1e7(EmpiricalBranchingFractionNoSpin):
    def __retention_file__(self):
        return os.path.join(os.path.dirname(__file__), "grid_dict_1e7")

class EmpiricalBranchingFractionNoSpin_1e8(EmpiricalBranchingFractionNoSpin):
    def __retention_file__(self):
        return os.path.join(os.path.dirname(__file__), "grid_dict_1e8")
    
def low_spin_component(spin):
    return xp.asarray(spin == 0).astype(float)


def low_spin_component_grid(spin):
    delta = xp.asarray(spin == 0).astype(float)
    return delta / trapz(delta, spin)


def first_generation_spin_magnitude(spin, alpha, beta, delta, a_max):
    beta_component = beta_dist(
        xx=spin, alpha=alpha, beta=beta, scale=a_max
    )
    beta_component[beta_component == NAN_INF] = 0
    return delta * low_spin_component(spin) + (1 - delta) * beta_component


def first_generation_spin_magnitude_grid(spin, alpha, beta, delta, a_max):
    beta_component = beta_dist(
        xx=spin, alpha=alpha, beta=beta, scale=a_max
    )
    beta_component[beta_component == NAN_INF] = 0
    return delta * low_spin_component_grid(spin) + (1 - delta) * beta_component


def two_component_primary_mass_ratio_dynamical_without_spins(
    dataset,
    alpha,
    beta,
    mmin,
    mmax,
    lam,
    mpp,
    sigpp,
    branch_1=0.12,
    branch_2=0.01,
):
    """
    Power law model for two-dimensional mass distribution, modelling primary
    mass and conditional mass ratio distribution.

    p(m1, q) = p(m1) * p(q | m1)

    We also include the effect of dynamical mergers leading to 1.5 and 2nd
    generation mergers.

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays for 'mass_1' and 'mass_ratio'.
    alpha: float
        Negative power law exponent for more massive black hole.
    mmin: float
        Minimum black hole mass.
    mmax: float
        Maximum black hole mass.
    beta: float
        Power law exponent of the mass ratio distribution.
    lam: float
        Fraction of black holes in the Gaussian component.
    mpp: float
        Mean of the Gaussian component.
    sigpp: float
        Standard deviation fo the Gaussian component.
    branch_1: float
        Ratio of 1.5 generation mergers to 1st generation mergers.
        The default value comes from a conversation with Eric Thrane.
    branch_2: float
        Ratio of 2nd generation mergers to 1st generation mergers.
        The default value comes from a conversation with Eric Thrane.
    """
    btot = 1 + branch_1 + branch_2

    fraction_1_5 = branch_1 / btot
    fraction_2 = branch_2 / btot
    fraction_1 = 1 - fraction_1_5 - fraction_2

    if fraction_1 < 0:
        return np.zeros_like(dataset["mass_1"])
    # assert branch_0 >= 0, "Branching fractions greater than 1."
    first_generation = two_component_primary_mass_ratio(
        dataset=dataset,
        alpha=alpha,
        beta=beta,
        mmin=mmin,
        mmax=mmax,
        lam=lam,
        mpp=mpp,
        sigpp=sigpp,
    )
    params = dict(
        mmin=mmin * 2, mmax=mmax * 2, lam=lam, mpp=mpp * 2, sigpp=sigpp * 2
    )
    one_point_five_generation = two_component_single(
        dataset["mass_1"], alpha=alpha, **params
    ) * one_point_five_generation_mass_ratio(
        dataset, spectal_index=beta * 1.5, mmin=mmin
    )
    second_generation = two_component_primary_mass_ratio(
        dataset=dataset,
        alpha=alpha,
        beta=beta * 4,
        mmin=mmin * 2,
        mmax=mmax * 2,
        lam=lam,
        mpp=mpp * 2,
        sigpp=sigpp * 2,
    )
    return (
        fraction_1 * first_generation
        + fraction_1_5 * one_point_five_generation
        + fraction_2 * second_generation
    )


def one_point_five_generation_mass_ratio(dataset, spectal_index, mmin):
    split = (1 + mmin / dataset["mass_1"]) / 2
    prob = (
        powerlaw(
            dataset["mass_ratio"],
            spectal_index,
            high=split,
            low=mmin / dataset["mass_1"],
        )
        * (dataset["mass_ratio"] <= split)
        + powerlaw(
            1 - dataset["mass_ratio"],
            spectal_index,
            high=split,
            low=mmin / dataset["mass_1"],
        )
        * (dataset["mass_ratio"] >= split)
    ) / 2
    return prob
