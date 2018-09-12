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
        self._draws = np.empty((num_params, num_samples_saved, num_chains), order="F")

        for chain_index, stan_output in zip(range(self.num_chains), self.stan_outputs):
            draw_index = 0
            for entry in stan_output:
                if entry["topic"] == "SAMPLE":
                    draw = []
                    # Check for a sample message which is mixed together with
                    # proper parameter samples.  Planned changes in the services
                    # API may make this check unnecessary.
                    if entry["feature"] and entry["feature"][0].get("name") is None:
                        continue

                    for value_wrapped in entry["feature"]:
                        # for now, skip things such as lp__, stepsize__
                        if value_wrapped["name"].endswith("__"):
                            continue
                        kind = "doubleList" if "doubleList" in value_wrapped else "intList"
                        # extract int or double depending on 'kind'
                        value = value_wrapped[kind]["value"].pop()
                        draw.append(value)
                    self._draws[:, draw_index, chain_index] = draw
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

    def __getitem__(self, param):
        """Returns array with shape (stan_dimensions, num_chains * num_samples)"""
        if not self._finished:
            raise RuntimeError("Still collecting draws for fit.")
        assert param in self.param_names, param
        param_indexes = self._parameter_indexes(param)
        param_dim = self.dims[self.param_names.index(param)]
        # fmt: off
        num_samples_saved = (self.num_samples + self.num_warmup * self.save_warmup) // self.num_thin
        assert self.values.shape == (len(self.constrained_param_names), num_samples_saved, self.num_chains)
        # fmt: on
        # Stack chains together. Parameter is still stored flat.
        view = self.values[param_indexes, :, :].reshape(len(param_indexes), -1).view()
        assert view.shape == (len(param_indexes), num_samples_saved * self.num_chains)
        # reshape must yield something with least two dimensions
        reshape_args = param_dim + [-1] if param_dim else (1, -1)
        # reshape, recover the shape of the stan parameter
        return view.reshape(*reshape_args, order="F")

    def _parameter_indexes(self, param: str) -> Sequence[int]:
        """Obtain indexes for values associated with `param`.

        A draw from the sampler is a flat vector of values. A multi-dimensional
        variable will be stored in this vector in column-major order. This function
        identifies the indices which allow us to extract values associated with a
        parameter.

        Parameters
        ----------
        param : Parameter of interest.

        Returns
        -------
        Indexes associated with parameter.
        """

        # if `param` is a scalar, it will match one of the constrained_names
        if param in self.constrained_param_names:
            return (self.constrained_param_names.index(param),)

        def calculate_starts(dims: Sequence[Sequence[int]]) -> Sequence[int]:
            """Calculate starting indexes given dims."""
            s = [np.prod(d) for d in dims]
            starts = np.cumsum([0] + s)[: len(dims)]
            return tuple(int(i) for i in starts)

        starts = calculate_starts(self.dims)
        names_index = self.param_names.index(param)
        flat_param_count = np.prod(self.dims[names_index])
        return tuple(starts[names_index] + offset for offset in range(flat_param_count))
