======
PyStan
======

**NOTE: This is an experimental version of PyStan 3. Do not use! If you must use this, install
it in a virtualenv so it will not conflict with Pystan 2.x.**

Python interface to Stan, a package for Bayesian inference.

Stan® is a state-of-the-art platform for statistical modeling and
high-performance statistical computation. Thousands of users rely on Stan for
statistical modeling, data analysis, and prediction in the social, biological,
and physical sciences, engineering, and business.

* Open source software: ISC License

Getting Started
===============

This is an example in Section 5.5 of Gelman et al (2003), which studied
coaching effects from eight schools. For simplicity, we call this example
"eight schools."

.. code-block:: python

    import stan
    program_code = """
    data {
      int<lower=0> J; // number of schools
      real y[J]; // estimated treatment effects
      real<lower=0> sigma[J]; // s.e. of effect estimates
    }
    parameters {
      real mu;
      real<lower=0> tau;
      real eta[J];
    }
    transformed parameters {
      real theta[J];
      for (j in 1:J)
        theta[j] = mu + tau * eta[j];
    }
    model {
      target += normal_lpdf(eta | 0, 1);
      target += normal_lpdf(y | theta, sigma);
    }
    """

    data = {'J': 8,
            'y': [28,  8, -3,  7, -1,  1, 18, 12],
            'sigma': [15, 10, 16, 11,  9, 11, 10, 18]}

    posterior = stan.build(program_code, data=data)
    fit = posterior.sample(num_chains=4, num_samples=1000)
    eta = fit["eta"]  # array with shape (8, 4000)
    df = fit.to_frame()  # pandas `DataFrame`


Installation
============

Install from source in a virtualenv.
