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
