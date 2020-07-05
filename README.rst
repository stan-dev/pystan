******
PyStan
******

**NOTE: This documentation describes a BETA release of PyStan 3.**

**PyStan** is a Python interface to Stan, a package for Bayesian inference.

Stan® is a state-of-the-art platform for statistical modeling and
high-performance statistical computation. Thousands of users rely on Stan for
statistical modeling, data analysis, and prediction in the social, biological,
and physical sciences, engineering, and business.

Notable features of PyStan include:

* Automatic caching of compiled Stan models
* Automatic caching of samples from Stan models
* An interface similar to that of RStan
* Open source software: ISC License

Getting started
===============

**NOTE: BETA versions of PyStan 3 must be installed with ``pip install --pre pystan``.**

Install PyStan with ``pip install pystan``.  (PyStan requires Python 3.7 or higher running on a Linux or macOS system.)

The following block of code shows how to use PyStan with a model which studied coaching effects across eight schools (see Section 5.5 of Gelman et al (2003)). This hierarchical model is often called the "eight schools" model.

.. code-block:: python

    import stan

    schools_code = """
    data {
      int<lower=0> J;         // number of schools
      real y[J];              // estimated treatment effects
      real<lower=0> sigma[J]; // standard error of effect estimates
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
    df = fit.to_frame()  # pandas `DataFrame`
