from stan.model import build  # noqa

try:
    from importlib.metadata import version

    __version__ = version("pystan")
except ModuleNotFoundError:
    pass
