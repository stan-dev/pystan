from typing import Any, Mapping, Sequence

import numpy as np


class Fit:
    """Stores draws from one or more chains.

    The `values` attribute provides direct access to draws. More user-friendly
    presentations of draws are available via the `to_frame` and `to_xarray`
    methods.

    Attributes:
        values: An ndarray with shape (num_chains, num_draws, num_flat_params)

    """

    # TODO: Several possible optimizations to be made:
    # (1) `Fit` could be built up (concurrently) one chain a time with an
    # `append` method. This could be significantly faster.
    # (2) `Fit` need not store full copies of the raw Stan output.
    def __init__(
        self,
        stan_outputs: Sequence[Sequence[Mapping[str, Any]]],
        num_chains: int,
        param_names: Sequence[str],
        constrained_param_names: Sequence[str],
        dims: Sequence[Sequence[int]],
        num_warmup: int,
        num_samples: int,
        num_thin: int,
        save_warmup: bool,
    ) -> None:
        self.stan_outputs = stan_outputs
        self.num_chains = num_chains
        assert self.num_chains == len(self.stan_outputs)
        self.param_names, self.dims, self.constrained_param_names = (
            param_names,
            dims,
            constrained_param_names,
        )
        self.num_warmup, self.num_samples = num_warmup, num_samples
        self.num_thin, self.save_warmup = num_thin, save_warmup

        num_params = sum(
            np.product(dims_ or 1) for dims_ in dims
        )  # if dims == [] then it is a scalar
        assert num_params == len(constrained_param_names), (num_params, constrained_param_names)
        num_samples_saved = (
            self.num_samples + self.num_warmup * self.save_warmup
        ) // self.num_thin
        # order is 'F' for 'Fortran', column-major order
        self._draws = np.empty((num_chains, num_samples_saved, num_params), order="F")

        for chain_index, stan_output in zip(range(self.num_chains), self.stan_outputs):
            draw_index = 0
            for entry in stan_output:
                if entry["topic"] == "SAMPLE":
                    # Check for a sample message which is mixed together with
                    # proper parameter samples.  Planned changes in the services
                    # API may make this check unnecessary.
                    if "" in entry["feature"]:
                        continue

                    draw = []
                    for key in entry["feature"].keys():
                        # for now, skip things such as lp__, stepsize__
                        if key.endswith("__"):
                            continue
                        value_wrapped = entry["feature"][key]
                        kind = "doubleList" if "doubleList" in value_wrapped else "intList"
                        # extract int or double depending on 'kind'
                        value = value_wrapped[kind]["value"].pop()
                        draw.append(value)
                    self._draws[chain_index, draw_index, :] = draw
                    draw_index += 1
            assert draw_index == num_samples_saved
        # set draws array to read-only, also indicates we are finished
        self._draws.flags["WRITEABLE"] = False

    def __contains__(self, key):
        if not self._finished:
            raise RuntimeError("Still collecting draws for fit.")
        return key in self.param_names

    def to_frame(self):
        """Return view of draws as a pandas DataFrame.

        If pandas is not installed, a `RuntimeError` will be raised.

        Returns:
            pandas.DataFrame: DataFrame with `num_draws` rows and
                `num_flat_params` columns.
        """
        try:
            import pandas as pd
        except ImportError as ex:
            raise RuntimeError("The `to_frame` method requires the Python package `pandas`.")
        constrained_param_names = self.constrained_param_names
        df = pd.DataFrame(
            self._draws.reshape(-1, len(constrained_param_names)), columns=constrained_param_names
        )
        df.index.name, df.columns.name = "draws", "parameters"
        return df

    @property
    def values(self):
        return self._draws

    @property
    def _finished(self):
        return not self._draws.flags["WRITEABLE"]
