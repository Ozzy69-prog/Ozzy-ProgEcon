import pandas as pd
import numpy as np
from IPython.display import display

import matplotlib.pyplot as plt

from dstapi import DstApi
from scipy.stats import norm

Seed = 2026


def human_capital_shock(rng, std, size):
    """Draw a mean-one lognormal shock to human capital.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator to draw from.
    std : float
        Standard deviation of the underlying normal distribution. If 0,
        no shock is applied (all draws are exactly 1).
    size : int
        Number of draws.

    Returns
    -------
    ndarray of shape (size,)
        Lognormal shocks psi with log(psi) ~ N(-0.5*std**2, std**2), so
        that E[psi] = 1.
    """
    if std == 0:
        return np.ones(size)
    return rng.lognormal(mean=-0.5 * std**2, sigma=std, size=size)


def update_human_capital(human_capital, growth, depreciation, psi, employed):
    """Update human capital by one period given employment status.

    hi,t+1 = hi,t * (1 + growth) * psi   if employed
    hi,t+1 = hi,t * (1 - depreciation) * psi   if unemployed

    Parameters
    ----------
    human_capital : ndarray
        Current human capital.
    growth : ndarray
        Education-specific human capital growth rate while employed.
    depreciation : float
        Human capital depreciation rate while unemployed.
    psi : ndarray
        Mean-one lognormal shock for this period.
    employed : ndarray of {0, 1}
        Employment state (1=employed, 0=unemployed).

    Returns
    -------
    ndarray
        Updated human capital.
    """
    return np.where(
        employed == 1,
        human_capital * (1 + growth) * psi,
        human_capital * (1 - depreciation) * psi,
    )


def simulate_income(
    N, Pe, Se, He, growth, depreciation, std,
    job_finding, job_separation, grant, replacement, benefit_floor,
    start_age, retire_age, seed,
    education_on=True, shocks_on=True, depreciation_on=True, unemployment_on=True,
):
    """Simulate the life-cycle labor market, human capital, and income process.

    A cohort of N individuals is followed from start_age to retire_age.
    Everybody starts in education, draws an education level (short/medium/
    long) with probabilities Pe, and enters the labor market once their
    education is finished. From then on, employment follows a two-state
    Markov chain (job_finding / job_separation). Human capital grows while
    employed and depreciates while unemployed, subject to a mean-one
    lognormal shock each period. Income depends on education/employment
    status: the student grant while in education, human capital while
    employed, a replacement rate times the last wage while unemployed (or
    a benefit floor for those who have never been employed).

    This single function covers both the baseline simulation (Task 2.2,
    all flags True) and the mechanism decomposition (Task 2.4): the four
    `*_on` flags let individual mechanisms be switched off one at a time.

    Parameters
    ----------
    N : int
        Number of individuals.
    Pe : sequence of float, length 3
        Probabilities of (short, medium, long) education. Must sum to 1.
    Se : sequence of float, length 3
        Years of education for (short, medium, long).
    He : sequence of float, length 3
        Initial human capital he,0 for (short, medium, long).
    growth : sequence of float, length 3
        Human capital growth rate while employed, for (short, medium, long).
    depreciation : float
        Human capital depreciation rate while unemployed.
    std : float
        Standard deviation of the human capital shock.
    job_finding : float
        Probability an unemployed person finds a job (lambda).
    job_separation : float
        Probability an employed person loses their job (sigma).
    grant : float
        Student grant received while in education.
    replacement : float
        Replacement rate applied to the last wage while unemployed.
    benefit_floor : float
        Income floor for those who have never been employed.
    start_age, retire_age : int
        First and last age simulated (inclusive).
    seed : int
        Seed for this simulation's own random number generator, so that
        repeated calls (e.g. across scenarios) are independent and
        reproducible.
    education_on : bool, default True
        If False, everyone is treated as a single "average" group with
        education-probability-weighted Se/He/growth, instead of drawing
        distinct education levels.
    shocks_on : bool, default True
        If False, the human capital shock is switched off (psi = 1).
    depreciation_on : bool, default True
        If False, human capital does not depreciate while unemployed.
    unemployment_on : bool, default True
        If False, everyone is employed immediately upon entering the labor
        market and stays employed (job_finding=1, job_separation=0).

    Returns
    -------
    states : ndarray, shape (N,)
        Employment state at retirement (1=employed, 0=unemployed).
    human_capital : ndarray, shape (N,)
        Human capital at retirement.
    income : ndarray, shape (N, T)
        Income for every individual at every simulated age.
    ages : ndarray, shape (T,)
        Ages corresponding to the columns of `income`.
    education_group : ndarray, shape (N,)
        0=short, 1=medium, 2=long education, or -1 for every individual if
        education_on=False.
    """
    rng = np.random.default_rng(seed)

    dep = depreciation if depreciation_on else 0.0
    sd = std if shocks_on else 0.0
    jf = job_finding if unemployment_on else 1.0
    js = job_separation if unemployment_on else 0.0

    He_arr = np.asarray(He, dtype=float)
    growth_arr = np.asarray(growth, dtype=float)
    Se_arr = np.asarray(Se, dtype=float)

    if education_on:
        draws = rng.random(N)
        # Vectorized version of: 0 if draw<=Pe[0], 1 if draw<=Pe[0]+Pe[1], else 2
        education_group = np.searchsorted(np.cumsum(Pe), draws, side="right")
        human_capital = He_arr[education_group]
        growth_vec = growth_arr[education_group]
        entry_age = start_age + Se_arr[education_group] + 1
    else:
        Se_avg = sum(p * s for p, s in zip(Pe, Se))
        He_avg = sum(p * h for p, h in zip(Pe, He))
        growth_avg = sum(p * g for p, g in zip(Pe, growth))
        education_group = np.full(N, -1)
        human_capital = np.full(N, He_avg)
        growth_vec = np.full(N, growth_avg)
        entry_age = np.full(N, start_age + round(Se_avg) + 1, dtype=float)

    states = np.zeros(N, dtype=int)  # 1=employed, 0=unemployed
    ages = np.arange(start_age, retire_age + 1)
    T = len(ages)
    income = np.zeros((N, T))

    ever_employed = np.zeros(N, dtype=bool)
    last_wage = np.zeros(N)

    for t, age in enumerate(ages):
        active_mask = age >= entry_age

        # Employment transitions (only for those already in the labor market)
        unemployed_active = active_mask & (states == 0)
        employed_active = active_mask & (states == 1)
        states[unemployed_active & (rng.random(N) < jf)] = 1
        states[employed_active & (rng.random(N) < js)] = 0

        education_mask = ~active_mask
        employed_mask = active_mask & (states == 1)
        unemployed_mask = active_mask & (states == 0)

        income[education_mask, t] = grant

        income[employed_mask, t] = human_capital[employed_mask]
        ever_employed[employed_mask] = True
        last_wage[employed_mask] = human_capital[employed_mask]

        income[unemployed_mask & ever_employed, t] = replacement * last_wage[unemployed_mask & ever_employed]
        income[unemployed_mask & ~ever_employed, t] = benefit_floor

        psi = human_capital_shock(rng, sd, N)
        human_capital[active_mask] = update_human_capital(
            human_capital[active_mask],
            growth_vec[active_mask],
            dep,
            psi[active_mask],
            states[active_mask],
        )

    return states, human_capital, income, ages, education_group


def gini_coefficient(income_vector):
    """Compute the Gini coefficient from a 1D income vector."""
    x = np.asarray(income_vector, dtype=float).ravel()
    x = x[np.isfinite(x)]  # Drop NaN/inf values.

    if x.size == 0:
        raise ValueError("income_vector must contain at least one finite value")
    if np.any(x < 0):
        raise ValueError("income_vector must be non-negative")

    total_income = x.sum()
    if np.isclose(total_income, 0.0):
        return 0.0

    x_sorted = np.sort(x)
    n = x_sorted.size
    rank = np.arange(1, n + 1)

    gini = (2 * np.sum(rank * x_sorted)) / (n * total_income) - (n + 1) / n
    return float(gini)


def lorenz_curve(income_vector):
    """Return cumulative population and income shares for a Lorenz curve."""
    x = np.asarray(income_vector, dtype=float).ravel()
    x = x[np.isfinite(x)]

    if x.size == 0:
        raise ValueError("income_vector must contain at least one finite value")
    if np.any(x < 0):
        raise ValueError("income_vector must be non-negative")

    x_sorted = np.sort(x)
    total_income = x_sorted.sum()
    n = x_sorted.size

    if np.isclose(total_income, 0.0):
        cum_income = np.zeros(n + 1)
    else:
        cum_income = np.insert(np.cumsum(x_sorted) / total_income, 0, 0.0)

    cum_population = np.linspace(0.0, 1.0, n + 1)
    return cum_population, cum_income
