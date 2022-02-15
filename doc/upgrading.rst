.. _upgrading:

===========================
Upgrading to Newer Releases
===========================

Upgrading to 3.0
================

PyStan 3 is a complete rewrite. PyStan runs on Linux and macOS.

PyStan 3 makes numerous **backwards-incompatible changes**.
Many of these changes are introduced to harmonize variable naming practices across the numerous interfaces to the Stan C++ library.

The scope of PyStan 3 is reduced in the interest of freeing up resources to dedicate to making the software more reliable and to guaranteeing timely releases. The default HMC sampler is supported. Variational inference, for example, is no longer supported.

PyStan 3 users aware of changes in variable, function, and method names should be able to upgrade without much difficulty. The basic programming "flow" is essentially unchanged.

Here's how we draw from the posterior distribution in the eight schools model using PyStan 3:

.. code-block:: python

    import stan

    schools_code = """data { ..."""
    schools_data = {'J': 8, ... }

    posterior = stan.build(schools_code, data=schools_data, random_seed=1)
    fit = posterior.sample(num_chains=4, num_samples=1000)
    fit["eta"]  # array with shape (8, 4000)

Compare this with similar PyStan 2 code:

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
- Data and random seed are provided earlier, to the ``build`` method. Previously these were provided before sampling.
- Use ``num_samples`` to indicate number of desired draws.
- Use ``fit["param"]`` instead of ``fit.extract()["param"]``. The shape of the array returned will be different.
- Draws are returned in a shape which reflects their shape in the Stan model. Number of draws is the trailing index.
- The "stansummary" display is no longer supported. Support for displaying effective sample size is planned.
- The ``check_hmc_diagnostics`` function is removed. Support for :ref:`plugins <plugins>` has been added to allow for the development of a replacement. The function was removed from PyStan because it is not part of the Stan C++ library.
- Microsoft Windows is not supported in PyStan 3. It was (partially) supported in PyStan 2.
- The default, recommended HMC sampler is fully supported. Variational inference, maximization algorithms, and other sampling algorithms are not supported. Users who need these features should consider using different software (e.g., CmdStan, CmdStanPy, jax, PyTorch).
