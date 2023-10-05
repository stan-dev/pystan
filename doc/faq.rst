============================
 Frequently Asked Questions
============================

How can I use PyStan with Jupyter Notebook or JupyterLab?
---------------------------------------------------------

Use `nest-asyncio <https://pypi.org/project/nest-asyncio/>`_. This package is needed
because Jupter Notebook blocks the use of certain `asyncio
<https://docs.python.org/3/library/asyncio.html>`_ functions. (To verify this, try
running ``asyncio.run(asyncio.sleep(1))`` in a notebook.) If you would like to learn
more about the problem, see the following issue: `ipython/ipykernel#548
<https://github.com/ipython/ipykernel/issues/548>`_. This problem only affects Jupyter
Notebook and derivatives. It does not affect IPython.

Is Windows supported?
---------------------

If `WSL2 <https://docs.microsoft.com/en-us/windows/wsl/>`_ works for you, then
the answer is "yes," otherwise the answer is "no."

Experience has proven that supporting PyStan on Windows is challenging and
requires a considerable investment of resources. If you would like to fund the
development and ongoing maintenance of a version of PyStan which works on
Windows, please post your proposal for discussion on the `Stan forum <https://discourse.mc-stan.org/>`_.

.. _faq_linux_distributions:

What Linux distributions are supported?
---------------------------------------

There are three officially supported Linux distributions:

- Debian 11
- Ubuntu 20.04
- Ubuntu 22.04

Users have reported that PyStan works on the following Linux distributions:

- Arch Linux 2021.05.01
- Fedora 34

How do I get parameter summary statistics?
------------------------------------------

Consider using the following::

    fit = posterior.sample(num_chains=4, num_samples=1000)
    df = fit.to_frame()
    print(df.describe().T)

The final line uses the `pandas.DataFrame.describe <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html>`_ method.
It should print useful statistics about the parameters::

                    count    mean     std     min     25%     50%     75%     max
    parameters
    lp__          4,000.0   -39.4     2.6   -48.4   -41.0   -39.2   -37.6   -32.7
    accept_stat__ 4,000.0     0.9     0.2     0.0     0.9     1.0     1.0     1.0
    stepsize__    4,000.0     0.3     0.0     0.3     0.3     0.3     0.3     0.3
    treedepth__   4,000.0     3.7     0.5     2.0     3.0     4.0     4.0     5.0
    n_leapfrog__  4,000.0    13.5     3.4     3.0    15.0    15.0    15.0    31.0
    divergent__   4,000.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0
    energy__      4,000.0    44.3     3.3    34.1    42.0    44.1    46.6    56.8
    mu            4,000.0     7.5     4.7    -9.5     4.6     7.6    10.7    24.6
    tau           4,000.0     6.5     5.3     0.0     2.4     5.4     9.1    36.8
    eta.1         4,000.0     0.4     0.9    -3.1    -0.2     0.4     1.0     3.5
    eta.2         4,000.0     0.0     0.9    -3.5    -0.5     0.0     0.6     2.7
    eta.3         4,000.0    -0.2     0.9    -3.9    -0.8    -0.2     0.4     3.0
    eta.4         4,000.0     0.0     0.9    -3.1    -0.5     0.0     0.6     3.5
    eta.5         4,000.0    -0.3     0.9    -3.1    -0.8    -0.3     0.3     2.8
    eta.6         4,000.0    -0.2     0.9    -3.4    -0.8    -0.2     0.3     2.7
    eta.7         4,000.0     0.3     0.9    -2.6    -0.3     0.4     0.9     3.1
    eta.8         4,000.0     0.0     0.9    -2.5    -0.5     0.0     0.6     2.6
    theta.1       4,000.0    11.0     8.1   -12.9     6.0    10.0    14.8    48.9
    theta.2       4,000.0     7.8     6.1   -14.9     3.8     7.6    11.5    29.5
    theta.3       4,000.0     6.1     7.7   -28.8     2.2     6.6    10.3    42.2
    theta.4       4,000.0     7.7     6.4   -15.0     3.7     7.5    11.6    30.2
    theta.5       4,000.0     5.2     6.1   -21.1     1.3     5.7     9.3    23.9
    theta.6       4,000.0     5.8     6.7   -24.3     1.4     6.2    10.1    28.0
    theta.7       4,000.0    10.4     6.5    -6.1     6.2    10.0    14.0    38.8
    theta.8       4,000.0     8.0     7.2   -18.7     3.8     7.9    12.1    35.1

Using this technique should give you much of the information provided
by the CmdStan program ``stansummary``.

How can I run pystan on macOS with Apple silicon chips (Apple M1, M2, etc)?
---------------------------------------------------------------------------

First, you have to install the httpstan package from source. This can be done
with the following steps:

1. Download source code from newest httpstan version from
   `httpstan's GitHub <https://github.com/stan-dev/httpstan/tags>`_.
2. Extract .zip or .tar.gz.
3. With an active virtual environment, navigate to the extracted folder in the
   terminal and follow instructions below taken from
   `httpstan's documentation <https://httpstan.readthedocs.io/en/latest/installation.html>`_:

   a. Build shared libraries by typing ``make`` into the terminal.
   b. Build the httpstan wheel on your system by typing ``python3 -m pip 
      install poetry`` followed by ``python3 -m poetry build`` into the
      terminal.
   c. Install the wheel by typing ``python3 -m pip install dist/*.whl`` into
      the terminal.

Second, you can install pystan from `PyPI <https://pypi.org/project/pystan/>`_
from source by typing ``pip install pystan --no-binary pystan`` into the
terminal.

You are now ready to use pystan and you can test it by running the "Quick
start" code provided
`here <https://pystan.readthedocs.io/en/latest/index.html>`_.
