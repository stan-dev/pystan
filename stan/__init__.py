import pbr.version

from stan.model import build  # noqa


__version__ = pbr.version.VersionInfo("stan").version_string()
