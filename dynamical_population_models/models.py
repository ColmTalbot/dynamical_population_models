from gwpopulation.conversions import mu_chi_var_chi_max_to_alpha_beta_max
from gwpopulation.utils import beta_dist, powerlaw
from gwpopulation.models.mass import (
    two_component_primary_mass_ratio, two_component_single)
from gwpopulation.models.spin import iid_spin_magnitude_beta


def two_component_primary_mass_ratio_dynamical_with_spins(
        dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp, branch_1=0.12,
        branch_2=0.01, mu_chi=0.67, var_chi=0.01):
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
        Fraction of 1.5 generation mergers.
        The default value comes from a conversation with Eric Thrane.
    branch_2: float
        Fraction of 2nd generation mergers.
        The default value comes from a conversation with Eric Thrane.
    """
    branch_0 = 1 - branch_1 - branch_2
    assert branch_0 >= 0, "Branching fractions greater than 1."
    first_generation_mass = two_component_primary_mass_ratio(
        dataset=dataset, alpha=alpha, beta=beta, mmin=mmin, mmax=mmax, lam=lam,
        mpp=mpp, sigpp=sigpp)
    params = dict(mmin=mmin * 2, mmax=mmax * 2, lam=lam, mpp=mpp * 2,
                  sigpp=sigpp * 2)
    one_point_five_generation_mass = (
            two_component_single(dataset['mass_1'], alpha=alpha, **params) *
            one_point_five_generation_mass_ratio(
                dataset, spectal_index=beta * 1.5, mmin=mmin))
    second_generation_mass = two_component_primary_mass_ratio(
        dataset=dataset, alpha=alpha, beta=beta * 2, mmin=mmin * 2,
        mmax=mmax * 2, lam=lam, mpp=mpp * 2, sigpp=sigpp * 2)

    first_generation_spin = (dataset["a_1"] == 0) & (dataset["a_2"] == 0)

    alpha, beta = mu_chi_var_chi_max_to_alpha_beta_max(
        mu_chi=mu_chi, var_chi=var_chi, amax=1)
    one_point_five_generation_spin = beta_dist(
        dataset["a_1"], scale=1, alpha=alpha, beta=beta) * (dataset["a_2"] == 0)
    second_generation_spin = iid_spin_magnitude_beta(
        dataset=dataset, alpha_chi=alpha, beta_chi=beta, amax=1)

    return (
        branch_0 * first_generation_mass * first_generation_spin +
        branch_1 * one_point_five_generation_mass * one_point_five_generation_spin +
        branch_2 * second_generation_mass * second_generation_spin)


def two_component_primary_mass_ratio_dynamical(
        dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp, branch_1=0.12,
        branch_2=0.01):
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
        Fraction of 1.5 generation mergers.
        The default value comes from a conversation with Eric Thrane.
    branch_2: float
        Fraction of 2nd generation mergers.
        The default value comes from a conversation with Eric Thrane.
    """
    branch_0 = 1 - branch_1 - branch_2
    assert branch_0 >= 0, "Branching fractions greater than 1."
    first_generation = two_component_primary_mass_ratio(
        dataset=dataset, alpha=alpha, beta=beta, mmin=mmin, mmax=mmax, lam=lam,
        mpp=mpp, sigpp=sigpp)
    params = dict(mmin=mmin * 2, mmax=mmax * 2, lam=lam, mpp=mpp * 2,
                  sigpp=sigpp * 2)
    one_point_five_generation = (
            two_component_single(dataset['mass_1'], alpha=alpha, **params) *
            one_point_five_generation_mass_ratio(
                dataset, spectal_index=beta * 1.5, mmin=mmin))
    second_generation = two_component_primary_mass_ratio(
        dataset=dataset, alpha=alpha, beta=beta * 2, mmin=mmin * 2,
        mmax=mmax * 2, lam=lam, mpp=mpp * 2, sigpp=sigpp * 2)
    return (
            branch_0 * first_generation +
            branch_1 * one_point_five_generation +
            branch_2 * second_generation)


def one_point_five_generation_mass_ratio(dataset, spectal_index, mmin):
    split = (1 + mmin / dataset["mass_1"]) / 2
    prob = (powerlaw(dataset['mass_ratio'], spectal_index, high=split,
                     low=mmin / dataset['mass_1'])
            * (dataset["mass_ratio"] <= split)
            + powerlaw(1 - dataset['mass_ratio'], spectal_index, high=split,
                       low=mmin / dataset['mass_1'])
            * (dataset["mass_ratio"] >= split)) / 2
    return prob


