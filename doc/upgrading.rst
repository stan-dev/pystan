.. _upgrading:

===========================
Upgrading to Newer Releases
===========================

.. caution::
    This section is a work-in-progress.

Upgrading to 3.0
================

**There are many backwards-incompatible changes in PyStan 3.0.**

PyStan 3 is a complete rewrite. Python 3.7 or higher is required. Linux and macOS are supported.

PyStan 3 and RStan 3 make numerous backwards-incompatible changes.
Many of these changes are introduced to harmonize variable naming practices across the numerous interfaces to the Stan C++ library.

The scope of PyStan 3 is reduced. The default HMC sampler is supported. Variational inference, for example, is no longer supported.

PyStan 3 users aware of changes in variable, function, and method names should be able to upgrade without much difficulty. The basic programming "flow" is essentially unchanged.

Here's how we draw from the posterior distribution in the eight schools model using PyStan 3:

.. code-block:: python

    import stan

    schools_code = """data { ..."""
    schools_data = {'J': 8, ... }

    posterior = stan.build(schools_code, data=schools_data, random_seed=1)
    fit = posterior.sample(num_chains=4, num_samples=1000)
    fit["eta"]  # array with shape (8, 4000)

Compare this with the equivalent PyStan 2 code:

.. code-block:: python

    import pystan

    schools_code = """data { ..."""
    schools_data = {'J': 8, ... }

    sm = pystan.StanModel(model_code=schools_code)
    fit = sm.sampling(data=schools_data, iter=1000, chains=4, seed=1)
    fit.extract()["eta"]  # array with shape (2000, 8)

Notable changes
---------------

- Use ``import stan`` instead of ``import pystan``.
- Data and random seed are provided earlier, in the build phase. Previously these were provided before sampling.
- Use ``num_samples`` to indicate number of desired draws.
- Use ``fit["param"]`` instead of ``fit.extract()["param"]``. The shape of the array returned will be different.
- Draws are returned in a shape which reflects their shape in the Stan model. Number of draws is the trailing index.
- The "stansummary" display is no longer supported. Effective sample size, however, is available.
- License is now ISC. Previously GPL3 was used.
- Microsoft Windows is not supported in PyStan 3. It was (partially) supported in PyStan 2.
- The default, recommended HMC sampler is fully supported. Variational inference, maximization algorithms, and other sampling algorithms are not supported. Users who need these features should consider using different software (e.g., CmdStan, CmdStanPy, jax, PyTorch).
