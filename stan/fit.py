from typing import Any, Sequence

import numpy as np

import httpstan.callbacks_writer_pb2 as callbacks_writer_pb2


class Fit:
    """Stores draws from one or more chains.

    The ``values`` attribute provides direct access to draws. More user-friendly
    presentations of draws are available via the ``to_frame`` and ``to_xarray``
    methods.

    Returned by methods of a ``Model``. Users will not instantiate this class directly.

    Attributes:
        values: An ndarray with shape (num_sample_and_sampler_params + num_flat_params, num_draws, num_chains)

    """

    # TODO: Several possible optimizations to be made:
    # (1) `Fit` could be built up (concurrently) one chain a time with an
    # `append` method. This could be significantly faster.
    # (2) `Fit` need not store full copies of the raw Stan output.
    def __init__(
        self,
        stan_outputs: Sequence[Sequence[Any]],
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

        # `self.sample_and_sampler_param_names` collects the sample and sampler param names.
        # - "sample params" include `lp__`, `accept_stat__`
        # - "sampler params" include `stepsize__`, `treedepth__`, ...
        # These names are gathered later in this function by inspecting the output from Stan.
        self.sample_and_sampler_param_names: Sequence[str]

        num_flat_params = sum(np.product(dims_ or 1) for dims_ in dims)  # if dims == [] then it is a scalar
        assert num_flat_params == len(constrained_param_names)
        num_samples_saved = (self.num_samples + self.num_warmup * self.save_warmup) // self.num_thin

        # self._draws holds all the draws. We cannot allocate it before looking at the draws
        # because we do not know how many sampler-specific parameters are present. Later in this
        # function we count them and only then allocate the array for `self._draws`.
        self._draws: np.ndarray

        for chain_index, stan_output in zip(range(self.num_chains), self.stan_outputs):
            draw_index = 0
            for msg in stan_output:
                if msg.topic == callbacks_writer_pb2.WriterMessage.Topic.Value("SAMPLE"):
                    # Ignore sample message which is mixed together with proper draws.
                    if msg.feature and msg.feature[0].name == "":
                        continue

                    draw_row = []  # a "row" of values from a single draw from Stan C++

                    # for the first draw: collect sample and sampler parameter names.
                    if not hasattr(self, "_draws"):
                        feature_names = tuple(fea.name for fea in msg.feature)
                        self.sample_and_sampler_param_names = tuple(
                            name for name in feature_names if name.endswith("__")
                        )
                        num_rows = len(self.sample_and_sampler_param_names) + num_flat_params
                        # column-major order ("F") aligns with how the draws are stored (in cols).
                        self._draws = np.empty((num_rows, num_samples_saved, num_chains), order="F")
                        # rudimentary check of parameter order (sample & sampler params must be first)
                        if num_flat_params and feature_names[-1].endswith("__"):
                            raise RuntimeError(
                                f"Expected last parameter name to be one declared in program code, found `{feature_names[-1]}`"
                            )

                    for fea in msg.feature:
                        value = (
                            fea.double_list.value.pop() if fea.HasField("double_list") else fea.int_list.value.pop()
                        )
                        draw_row.append(value)

                    self._draws[:, draw_index, chain_index] = draw_row
                    draw_index += 1
            assert draw_index == num_samples_saved
        # set draws array to read-only, also indicates we are finished
        assert self.sample_and_sampler_param_names and self._draws.size
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
        except ImportError:
            raise RuntimeError("The `to_frame` method requires the Python package `pandas`.")
        columns = self.sample_and_sampler_param_names + self.constrained_param_names
        assert len(self._draws) == len(columns)
        df = pd.DataFrame(self._draws.reshape(len(columns), -1).T, columns=columns)
        df.index.name, df.columns.name = "draws", "parameters"
        return df

    @property
    def values(self) -> np.ndarray:
        return self._draws

    @property
    def _finished(self) -> bool:
        return not self._draws.flags["WRITEABLE"]

    def __getitem__(self, param):
        """Returns array with shape (stan_dimensions, num_chains * num_samples)"""
        if not self._finished:
            raise RuntimeError("Still collecting draws for fit.")
        assert param.endswith("__") or param in self.param_names, param
        param_indexes = self._parameter_indexes(param)
        param_dim = [] if param in self.sample_and_sampler_param_names else self.dims[self.param_names.index(param)]
        # fmt: off
        num_samples_saved = (self.num_samples + self.num_warmup * self.save_warmup) // self.num_thin
        assert self.values.shape == (len(self.sample_and_sampler_param_names) + len(self.constrained_param_names), num_samples_saved, self.num_chains)
        # fmt: on
        # Stack chains together. Parameter is still stored flat.
        view = self.values[param_indexes, :, :].reshape(len(param_indexes), -1).view()
        assert view.shape == (len(param_indexes), num_samples_saved * self.num_chains)
        # reshape must yield something with least two dimensions
        reshape_args = param_dim + [-1] if param_dim else (1, -1)
        # reshape, recover the shape of the stan parameter
        return view.reshape(*reshape_args, order="F")

    def __len__(self) -> int:
        return len(self.param_names)

    def __repr__(self) -> str:
        # inspired by xarray
        parts = [f"<stan.{type(self).__name__}>"]

        def summarize_param(param_name, dims):
            return f"    {param_name}: {tuple(dims)}"

        if self.param_names:
            parts.append("Parameters:")
        for param_name, dims in zip(self.param_names, self.dims):
            parts.append(summarize_param(param_name, dims))
        if self._finished:
            # total draws is num_draws (per-chain) times num_chains
            parts.append(f"Draws: {self._draws.shape[-2] * self._draws.shape[-1]}")
        return "\n".join(parts)

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

        Note
        ----

        This function assumes that parameters appearing in the program code follow
        the sample and sampler parameters (e.g., ``lp__``, ``stepsize__``).
        """
        # if `param` is a scalar, it will match one of the constrained names or it will match a
        # sample param name (e.g., `lp__`) or a sampler param name (e.g., `stepsize__`)
        if param in self.sample_and_sampler_param_names:
            return (self.sample_and_sampler_param_names.index(param),)
        sample_and_sampler_params_offset = len(self.sample_and_sampler_param_names)
        if param in self.constrained_param_names:
            return (sample_and_sampler_params_offset + self.constrained_param_names.index(param),)

        def calculate_starts(dims: Sequence[Sequence[int]]) -> Sequence[int]:
            """Calculate starting indexes given dims."""
            s = [np.prod(d) for d in dims]
            starts = np.cumsum([0] + s)[: len(dims)]
            return tuple(int(i) for i in starts)

        starts = tuple(sample_and_sampler_params_offset + i for i in calculate_starts(self.dims))
        names_index = self.param_names.index(param)
        flat_param_count = np.prod(self.dims[names_index])
        return tuple(starts[names_index] + offset for offset in range(flat_param_count))
