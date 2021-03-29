import abc
from typing import Generator

import pkg_resources

import stan.fit


def get_plugins() -> Generator[pkg_resources.EntryPoint, None, None]:
    """Iterate over available plugins."""
    return pkg_resources.iter_entry_points(group="stan.plugins")


class PluginBase(abc.ABC):
    """Base class for PyStan plugins.

    Plugin developers should create a class which subclasses `PluginBase`.
    This class must be referenced in their package's entry points section.

    """

    # Implementation note: this plugin system is simple because there are only
    # a couple of places a plugin developer might want to change behavior. For
    # a more full-featured plugin system, see Stevedore
    # (<https://docs.openstack.org/stevedore>).  This plugin system follows
    # (approximately) the pattern stevedore labels `ExtensionManager`.

    def on_post_sample(self, fit: stan.fit.Fit) -> stan.fit.Fit:
        """Called with Fit instance when sampling has finished.

        The plugin can report information about the samples
        contained in the Fit object. It may also add to or
        modify the Fit instance.

        If the plugin only analyzes the contents of `fit`,
        it must return the `fit`.

        Argument:
            fit: Fit instance.

        Returns:
            The Fit instance.
        """
        return fit
