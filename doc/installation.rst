============
Installation
============

In order to install PyStan from PyPI make sure your system satisfies the requirements:

- Python ≥3.7
- Linux or macOS
- x86-64 CPU
- C++ compiler: gcc ≥9.0 or clang ≥10.0.

Install PyStan with ``pip``. The following command will install PyStan::

    python3 -m pip install pystan

Supported Linux distributions
-----------------------------

There are two officially supported Linux distributions:

- Debian 11
- Ubuntu 20.04

Users have reported that PyStan works on :ref:`several other Linux distributions <faq_linux_distributions>`.
If you use an unsupported distribution and find that the PyPI wheels do not work, try `installing
httpstan from source <https://httpstan.readthedocs.io/en/latest/installation.html>`_. Once httpstan
is installed, PyStan should work.
