.. _plugins:

=========
 Plugins
=========

This is a guide to installing and creating plugins for PyStan.

Installing Plugins
==================

In order to use a plugin, you need to install it. Plugins are published on PyPI and can be installed with ``pip``.

Plugins are automatically enabled as soon as they are installed.

Creating Plugins
================

Plugin developers should create a class which subclasses :py:class:`stan.plugins.PluginBase`. This
class must be referenced in their package's entry points section.

For example, if the class is ``mymodule.PrintParameterNames`` then the
setuptools configuration would look like the following::

    entry_points = {
      "stan.plugins": [
        "names = mymodule:PrintParameterNames"
      ]
    }

The equivalent configuration in poetry would be::

    [tool.poetry.plugins."stan.plugins"]
    names = mymodule:PrintParameterNames

You can define multiple plugins in the entry points section.  Note that the
plugin name (here, `names`) is required but is unused.

All :py:class:`stan.plugins.PluginBase` subclasses implement methods which define behavior associated with *events*.
There is only one event supported, ``post_sample``.

on_post_sample
--------------

This method defines what happens when sampling has finished and a
:py:class:`stan.fit.Fit` object is about to be returned to the user.  The
method takes a :py:class:`stan.fit.Fit` instance as an argument. The method
returns the instance. In a plugin, this method will typically analyze the data contained in
the instance. A plugin might also use this method to modify the instance, adding an
additional method or changing the behavior or an existing method.

**Arguments:**

- ``fit``: :py:class:`stan.fit.Fit` instance

For example, if you wanted to print the names of parameters you would define a plugin as follows::

    class PrintParameterNames(stan.plugins.PluginBase):
        def on_post_sample(self, fit, **kwargs):
            for key in fit:
                print(key)
            return fit

Note that `on_post_sample` accepts additional keyword arguments (``**kwargs``). Accepting
keyword arguments like this will allow your plugin to be compatible with future versions of the package.
Future versions of the package could, in principle, add additional arguments to `on_post_sample`.
