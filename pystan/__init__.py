import pbr.version

from pystan.model import build  # noqa


__version__ = pbr.version.VersionInfo("pystan").version_string()
