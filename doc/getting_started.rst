================
Getting Started
================

.. caution::
    This is a work-in-progress.

The following block of code shows how to use PyStan with a model which studied coaching effects across eight schools (see Section 5.5 of Gelman et al (2003)). This hierarchical model is often called the "eight schools" model.

Every Stan model starts with Stan program code. Begin by assigning the program code to the variable ``schools_code``.

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

Like most Stan models, this model references observations. Assign the data to the dictionary ``schools_data``.

.. code-block:: python

    schools_data = {"J": 8,
                    "y": [28,  8, -3,  7, -1,  1, 18, 12],
                    "sigma": [15, 10, 16, 11,  9, 11, 10, 18]}


Fitting a model using PyStan takes two steps. First we build the model using :py:func:`stan.build`.

.. code-block:: python

    posterior = stan.build(schools_code, data=schools_data, random_seed=1)

This function returns an instance of :py:class:`stan.model.Model`. (For reproducibility,
we specify a random seed using the `random_seed` argument.) Building, in this context, involves
converting the Stan program code into C++ code and then compiling that C++ code. This step may take some time.

Now we draw samples using the method :py:func:`stan.model.Model.sample`.
By setting `num_chains` to 4, we will draw samples in parallel using four CPU cores.

.. code-block:: python

    fit = posterior.sample(num_chains=4, num_samples=1000)

This method returns an instance of :py:func:`stan.fit.Fit`. This instance holds everything produced by the Stan sampler.
We can extract draws associated with a single variable using the familiar Python syntax.

.. code-block:: python

    eta = fit["eta"]  # array with shape (8, 4000)

Alternatively, we can extract all variables into a pandas DataFrame.

.. code-block:: python

    df = fit.to_frame()

Using the ``to_frame()`` method requires pandas. (Installing ``pystan`` will not install ``pandas``.) Install pandas with ``python3 -m pip install pandas``.
