******
PyStan
******

Release v\ |version|

**PyStan** is a Python interface to Stan, a package for Bayesian inference.

Stan® is a state-of-the-art platform for statistical modeling and
high-performance statistical computation. Thousands of users rely on Stan for
statistical modeling, data analysis, and prediction in the social, biological,
and physical sciences, engineering, and business.

Notable features of PyStan include:

* Automatic caching of compiled Stan models
* Automatic caching of samples from Stan models
* Open source software: ISC License

(Upgrading from PyStan 2? We have you covered: :ref:`upgrading`.)

.. _getting-started:

Quick start
===========

Install PyStan with ``python3 -m pip install pystan``. PyStan runs on Linux and macOS. You will also need a C++ compiler such as gcc ≥9.0 or clang ≥10.0.

This block of code shows how to use PyStan with a hierarchical model used to study coaching effects across eight schools (see Section 5.5 of Gelman et al. (2003)).

.. code-block:: python

    import stan

    schools_code = """
    data {
      int<lower=0> J;         // number of schools
      array[J] real y;              // estimated treatment effects
      array[J] real<lower=0> sigma; // standard error of effect estimates
    }
    parameters {
      real mu;                // population treatment effect
      real<lower=0> tau;      // standard deviation in treatment effects
      vector[J] eta;          // unscaled deviation from mu by school
    }
    transformed parameters {
      vector[J] theta = mu + tau * eta;        // school treatment effects
    }
    model {
      target += normal_lpdf(eta | 0, 1);       // prior log-density
      target += normal_lpdf(y | theta, sigma); // log-likelihood
    }
    """

    schools_data = {"J": 8,
                    "y": [28,  8, -3,  7, -1,  1, 18, 12],
                    "sigma": [15, 10, 16, 11,  9, 11, 10, 18]}

    posterior = stan.build(schools_code, data=schools_data)
    fit = posterior.sample(num_chains=4, num_samples=1000)
    eta = fit["eta"]  # array with shape (8, 4000)
    df = fit.to_frame()  # pandas `DataFrame, requires pandas


Documentation
=============

.. toctree::
   :maxdepth: 1

   getting_started
   installation
   upgrading
   reference
   plugins
   contributing
   faq
