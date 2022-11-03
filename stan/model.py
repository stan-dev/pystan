import asyncio
import dataclasses
import json
import re
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union, cast

import httpstan.models
import httpstan.schemas
import httpstan.services.arguments as arguments
import httpstan.utils
import numpy as np
import simdjson
from clikit.io import ConsoleIO

import stan.common
import stan.fit
import stan.plugins

Data = Dict[str, Union[int, float, Sequence[Union[int, float]]]]


class DataJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # numpy.ndarray is *unofficially* supported as there is no easy way to
        # construct tabular data using the Python standard library.
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # unofficially support np.int64, np.int32, etc.
        if hasattr(obj, "dtype") and np.issubdtype(obj.dtype, np.integer):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


# superficial frozendict implementation. Only used for function signatures
class frozendict(dict):
    def __setitem__(self, key, value):
        raise TypeError("'frozendict' object is immutable.")


@dataclasses.dataclass(frozen=True)
class Model:
    """Stores data associated with and proxies calls to a Stan model.

    Returned by ``build``. Users will not instantiate this class directly.

    """

    model_name: str
    program_code: str
    data: Data
    param_names: Tuple[str, ...]
    constrained_param_names: Tuple[str, ...]
    dims: Tuple[Tuple[int, ...]]
    random_seed: Optional[int]

    def __post_init__(self):
        if self.model_name != httpstan.models.calculate_model_name(self.program_code):
            raise ValueError("`model_name` does not match `program_code`.")

    def sample(self, *, num_chains=4, **kwargs) -> stan.fit.Fit:
        """Draw samples from the model.

        Parameters in ``kwargs`` will be passed to the default sample function.
        The default sample function is currently
        ``stan::services::sample::hmc_nuts_diag_e_adapt``.  Parameter names are
        identical to those used in CmdStan.  See the CmdStan documentation for
        parameter descriptions and default values.

        There is one exception:  `num_chains`. `num_chains` is a
        PyStan-specific keyword argument. It indicates the number of
        independent processes to use when drawing samples.


        Returns:
            Fit: instance of Fit allowing access to draws.

        Examples:
            User-defined initial values for parameters must be provided
            for each chain. Typically they will be the same for each chain.
            The following example shows how user-defined initial parameters
            are provided:

            >>> program_code = "parameters {real y;} model {y ~ normal(0,1);}"
            >>> posterior = stan.build(program_code)
            >>> fit = posterior.sample(num_chains=2, init=[{"y": 3}, {"y": 3}])

        """
        return self.hmc_nuts_diag_e_adapt(num_chains=num_chains, **kwargs)

    def hmc_nuts_diag_e_adapt(self, *, num_chains=4, **kwargs) -> stan.fit.Fit:
        """Draw samples from the model using ``stan::services::sample::hmc_nuts_diag_e_adapt``.

        Parameters in ``kwargs`` will be passed to the (Python wrapper of)
        ``stan::services::sample::hmc_nuts_diag_e_adapt``. Parameter names are
        identical to those used in CmdStan.  See the CmdStan documentation for
        parameter descriptions and default values.

        There is one exception:  `num_chains`. `num_chains` is a
        PyStan-specific keyword argument. It indicates the number of
        independent processes to use when drawing samples.

        Returns:
            Fit: instance of Fit allowing access to draws.

        """
        function = "stan::services::sample::hmc_nuts_diag_e_adapt"
        return self._create_fit(function=function, num_chains=num_chains, **kwargs)

    def fixed_param(self, *, num_chains=4, **kwargs) -> stan.fit.Fit:
        """Draw samples from the model using ``stan::services::sample::fixed_param``.

        Parameters in ``kwargs`` will be passed to the (Python wrapper of)
        ``stan::services::sample::fixed_param``. Parameter names are
        identical to those used in CmdStan.  See the CmdStan documentation for
        parameter descriptions and default values.

        There is one exception:  `num_chains`. `num_chains` is a
        PyStan-specific keyword argument. It indicates the number of
        independent processes to use when drawing samples.

        Returns:
            Fit: instance of Fit allowing access to draws.

        """
        function = "stan::services::sample::fixed_param"
        return self._create_fit(function=function, num_chains=num_chains, **kwargs)

    def _create_fit(self, *, function, num_chains, **kwargs) -> stan.fit.Fit:
        """Make a request to httpstan's ``create_fit`` endpoint and process results.

        Users should not use this function.

        Parameters in ``kwargs`` will be passed to the (Python wrapper of)
        `function`. Parameter names are identical to those used in CmdStan.
        See the CmdStan documentation for parameter descriptions and default
        values.

        Returns:
            Fit: instance of Fit allowing access to draws.

        """
        assert "chain" not in kwargs, "`chain` id is set automatically."
        assert "data" not in kwargs, "`data` is set in `build`."
        assert "random_seed" not in kwargs, "`random_seed` is set in `build`."

        # copy kwargs and verify everything is JSON-encodable
        kwargs = json.loads(DataJSONEncoder().encode(kwargs))

        # FIXME: special handling here for `init`, consistent with PyStan 2 but needs docs
        init: List[Data] = kwargs.pop("init", [dict() for _ in range(num_chains)])
        if len(init) != num_chains:
            raise ValueError("Initial values must be provided for each chain.")

        payloads = []
        for chain in range(1, num_chains + 1):
            payload = kwargs.copy()
            payload["function"] = function
            payload["chain"] = chain  # type: ignore
            payload["data"] = self.data  # type: ignore
            payload["init"] = init.pop(0)
            if self.random_seed is not None:
                payload["random_seed"] = self.random_seed  # type: ignore

            # fit needs to know num_samples, num_warmup, num_thin, save_warmup
            # progress reporting needs to know some of these
            num_warmup = payload.get("num_warmup", arguments.lookup_default(arguments.Method["SAMPLE"], "num_warmup"))
            num_samples = payload.get(
                "num_samples",
                arguments.lookup_default(arguments.Method["SAMPLE"], "num_samples"),
            )
            num_thin = payload.get("num_thin", arguments.lookup_default(arguments.Method["SAMPLE"], "num_thin"))
            save_warmup = payload.get(
                "save_warmup",
                arguments.lookup_default(arguments.Method["SAMPLE"], "save_warmup"),
            )
            payloads.append(payload)

        async def go():
            io = ConsoleIO()
            sampling_output = io.section().error_output
            percent_complete = 0
            sampling_output.write_line(f"<comment>Sampling:</comment> {percent_complete:3.0f}%")

            current_and_max_iterations_re = re.compile(r"Iteration:\s+(\d+)\s+/\s+(\d+)")
            async with stan.common.HttpstanClient() as client:
                operations = []
                for payload in payloads:
                    resp = await client.post(f"/{self.model_name}/fits", json=payload)
                    if resp.status == 422:
                        raise ValueError(str(resp.json()))
                    elif resp.status != 201:
                        raise RuntimeError(resp.json()["message"])
                    assert resp.status == 201
                    operations.append(resp.json())

                # poll to get progress for each chain until all chains finished
                current_iterations = {}
                while not all(operation["done"] for operation in operations):
                    for operation in operations:
                        if operation["done"]:
                            continue
                        resp = await client.get(f"/{operation['name']}")
                        assert resp.status != 404
                        operation.update(resp.json())
                        progress_message = operation["metadata"].get("progress")
                        if not progress_message:
                            continue
                        iteration, iteration_max = map(
                            int, current_and_max_iterations_re.findall(progress_message).pop(0)
                        )
                        if current_iterations.get(operation["name"]) == iteration:
                            continue
                        current_iterations[operation["name"]] = iteration
                        iterations_count = sum(current_iterations.values())
                        total_iterations = iteration_max * num_chains
                        percent_complete = 100 * iterations_count / total_iterations
                        sampling_output.clear() if io.supports_ansi() else sampling_output.write("\n")
                        sampling_output.write_line(
                            f"<comment>Sampling:</comment> {round(percent_complete):3.0f}% ({iterations_count}/{total_iterations})"
                        )
                    await asyncio.sleep(0.05)

                fit_in_cache = len(current_iterations) < num_chains

                stan_outputs = []
                for operation in operations:
                    fit_name = operation["result"].get("name")
                    if fit_name is None:  # operation["result"] is an error
                        assert not str(operation["result"]["code"]).startswith("2"), operation
                        message = operation["result"]["message"]
                        if """ValueError('Initialization failed.')""" in message:
                            sampling_output.clear()
                            sampling_output.write_line("<info>Sampling:</info> <error>Initialization failed.</error>")
                            raise RuntimeError("Initialization failed.")
                        raise RuntimeError(message)

                    resp = await client.get(f"/{fit_name}")
                    if resp.status != 200:
                        raise RuntimeError((resp.json())["message"])
                    stan_outputs.append(resp.content)

                    # clean up after ourselves when fit is uncacheable (no random seed)
                    if self.random_seed is None:
                        resp = await client.delete(f"/{fit_name}")
                        if resp.status not in {200, 202, 204}:
                            raise RuntimeError((resp.json())["message"])

                sampling_output.clear() if io.supports_ansi() else sampling_output.write("\n")
                sampling_output.write_line(
                    "<info>Sampling:</info> 100%, done."
                    if fit_in_cache
                    else f"<info>Sampling:</info> {percent_complete:3.0f}% ({iterations_count}/{total_iterations}), done."
                )
                if not io.supports_ansi():
                    sampling_output.write("\n")

            stan_outputs = tuple(stan_outputs)  # Fit constructor expects a tuple.

            def is_nonempty_logger_message(msg: simdjson.Object):
                return msg["topic"] == "logger" and msg["values"][0] != "info:"  # type: ignore

            def is_iteration_or_elapsed_time_logger_message(msg: simdjson.Object):
                # Assumes `msg` is a message with topic `logger`.
                text = msg["values"][0]  # type: ignore
                text = cast(str, text)
                return (
                    text.startswith("info:Iteration:")
                    or text.startswith("info: Elapsed Time:")
                    # this detects lines following "Elapsed Time:", part of a multi-line Stan message
                    or text.startswith("info:" + " " * 15)
                )

            parser = simdjson.Parser()
            nonstandard_logger_messages = []
            for stan_output in stan_outputs:
                for line in stan_output.splitlines():
                    # Do not attempt to parse non-logger messages. Draws could contain nan or inf values.
                    # simdjson cannot parse lines containing such values.
                    if b'"logger"' not in line:
                        continue
                    msg = parser.parse(line)
                    if is_nonempty_logger_message(msg) and not is_iteration_or_elapsed_time_logger_message(msg):
                        nonstandard_logger_messages.append(msg.as_dict())
                    del msg
            del parser  # simdjson.Parser is no longer used at this point.

            if nonstandard_logger_messages:
                io.error_line("<comment>Messages received during sampling:</comment>")
                for msg in nonstandard_logger_messages:
                    text = msg["values"][0].replace("info:", "  ").replace("error:", "  ")
                    if text.strip():
                        io.error_line(f"{text}")

            fit = stan.fit.Fit(
                stan_outputs,
                num_chains,
                self.param_names,
                self.constrained_param_names,
                self.dims,
                num_warmup,
                num_samples,
                num_thin,
                save_warmup,
            )

            for entry_point in stan.plugins.get_plugins():
                Plugin = entry_point.load()
                fit = Plugin().on_post_sample(fit)
            return fit

        try:
            return asyncio.run(go())
        except KeyboardInterrupt:
            return  # type: ignore

    def constrain_pars(
        self, unconstrained_parameters: Sequence[float], include_tparams: bool = True, include_gqs: bool = True
    ) -> Sequence[float]:
        """Transform a sequence of unconstrained parameters to their defined support,
           optionally including transformed parameters and generated quantities.

        Arguments:
            unconstrained_parameters: A sequence of unconstrained parameters.
            include_tparams: Boolean to control whether we include transformed parameters.
            include_gqs: Boolean to control whether we include generated quantities.

        Returns:
            A sequence of constrained parameters, optionally including transformed parameters.

        Note:
            The unconstrained parameters are passed to the `write_array` method of the `model_base`
            instance. See `model_base.hpp` in the Stan C++ library for details.
        """
        payload = {
            "data": self.data,
            "unconstrained_parameters": unconstrained_parameters,
            "include_tparams": include_tparams,
            "include_gqs": include_gqs,
        }

        async def go():
            async with stan.common.HttpstanClient() as client:
                resp = await client.post(f"/{self.model_name}/write_array", json=payload)
                if resp.status != 200:
                    raise RuntimeError(resp.json())
                return resp.json()["params_r_constrained"]

        return asyncio.run(go())

    def unconstrain_pars(self, constrained_parameters: Sequence[float]) -> Sequence[float]:
        """Reads constrained parameter values from their specified context and returns a
           sequence of unconstrained parameter values.

        Arguments:
            constrained_parameters: Constrained parameter values and their specified context

        Returns:
            A sequence of unconstrained parameters.

        Note:
            The unconstrained parameters are passed to the `transform_inits` method of the
            `model_base` instance. See `model_base.hpp` in the Stan C++ library for details.
        """
        payload = {"data": self.data, "constrained_parameters": constrained_parameters}

        async def go():
            async with stan.common.HttpstanClient() as client:
                resp = await client.post(f"/{self.model_name}/transform_inits", json=payload)
                if resp.status != 200:
                    raise RuntimeError(resp.json())
                return resp.json()["params_r_unconstrained"]

        return asyncio.run(go())

    def log_prob(self, unconstrained_parameters: Sequence[float], adjust_transform: bool = True) -> float:
        """Calculate the log probability of a set of unconstrained parameters.

        Arguments:
            unconstrained_parameters: A sequence of unconstrained parameters.
            adjust_transform: Apply jacobian adjust transform.

        Returns:
            The log probability of the unconstrained parameters.

        Notes:
            The unconstrained parameters are passed to the log_prob
            function in stan::model.

        """
        payload = {
            "data": self.data,
            "unconstrained_parameters": unconstrained_parameters,
            "adjust_transform": adjust_transform,
        }

        async def go():
            async with stan.common.HttpstanClient() as client:
                resp = await client.post(f"/{self.model_name}/log_prob", json=payload)
                if resp.status != 200:
                    raise RuntimeError(resp.json())
                return resp.json()["log_prob"]

        return asyncio.run(go())

    def grad_log_prob(self, unconstrained_parameters: Sequence[float]) -> float:
        """Calculate the gradient of the log posterior evaluated at
           the unconstrained parameters.

        Arguments:
            unconstrained_parameters: A sequence of unconstrained parameters.

        Returns:
            The gradient of the log posterior evaluated at the
            unconstrained parameters.

        Notes:
            The unconstrained parameters are passed to the log_prob_grad
            function in stan::model.
        """
        payload = {
            "data": self.data,
            "unconstrained_parameters": unconstrained_parameters,
        }

        async def go():
            async with stan.common.HttpstanClient() as client:
                resp = await client.post(f"/{self.model_name}/log_prob_grad", json=payload)
                if resp.status != 200:
                    raise RuntimeError(resp.json())
                return resp.json()["log_prob_grad"]

        return asyncio.run(go())


def build(program_code: str, data: Data = frozendict(), random_seed: Optional[int] = None) -> Model:
    """Build (compile) a Stan program.

    Arguments:
        program_code: Stan program code describing a Stan model.
        data: A Python dictionary or mapping providing the data for the
            model. Variable names are the keys and the values are their
            associated values. Default is an empty dictionary, suitable
            for Stan programs with no `data` block.
        random_seed: Random seed, a positive integer for random number
            generation. Used to make sure that results can be reproduced.

    Returns:
        Model: an instance of Model

    Notes:
        C++ reserved words and Stan reserved words may not be used for
        variable names; see the Stan User's Guide for a complete list.

    """
    # `data` must be JSON-serializable in order to send to httpstan
    data = json.loads(DataJSONEncoder().encode(data))

    async def go():
        io = ConsoleIO()
        # hack: use stdout instead of stderr because httpstan silences stderr during compilation
        building_output = io.section().output
        if not io.supports_ansi():
            building_output.write("<comment>Building...</comment>")
        async with stan.common.HttpstanClient() as client:
            # Check to see if model is in cache.
            model_name = httpstan.models.calculate_model_name(program_code)
            resp = await client.post(f"/{model_name}/params", json={"data": data})
            model_in_cache = resp.status != 404

            task = asyncio.create_task(client.post("/models", json={"program_code": program_code}))
            start = time.time()
            while True:
                done, pending = await asyncio.wait({task}, timeout=0.1)
                if done:
                    break
                if io.supports_ansi():
                    building_output.clear()
                    building_output.write(f"<comment>Building:</comment> {time.time() - start:0.1f}s")
            building_output.clear() if io.supports_ansi() else building_output.write("\n")
            # now that httpstan has released stderr, we can use error_output again
            building_output = io.section().error_output
            resp = task.result()

            if resp.status != 201:
                match = re.search(r"""ValueError\(['"](.*)['"]\)""", resp.json()["message"])
                if not match:  # unknown error, should not happen
                    raise RuntimeError(resp.json()["message"])
                exception_body = match.group(1).encode().decode("unicode_escape")
                error_type_match = re.match(r"(Semantic|Syntax) error", exception_body)
                if error_type_match:
                    error_type = error_type_match.group(0)
                    exception_body_without_first_line = exception_body.split("\n", 1)[1]
                    building_output.write_line(f"<info>Building:</info> <error>{error_type}:</error>")
                    building_output.write_line(f"<error>{exception_body_without_first_line}</error>")
                    raise ValueError(error_type)
                else:
                    raise RuntimeError(exception_body)
            building_output.clear() if io.supports_ansi() else building_output.write("\n")
            if model_in_cache:
                building_output.write("<info>Building:</info> found in cache, done.")
            else:
                building_output.write(f"<info>Building:</info> {time.time() - start:0.1f}s, done.")
            assert model_name == resp.json()["name"]
            if resp.json().get("stanc_warnings"):
                io.error_line("<comment>Messages from <fg=cyan;options=bold>stanc</>:</comment>")
                io.error_line(resp.json()["stanc_warnings"])

            resp = await client.post(f"/{model_name}/params", json={"data": data})
            if resp.status != 200:
                raise RuntimeError(resp.json()["message"])
            params_list = resp.json()["params"]
            assert len({param["name"] for param in params_list}) == len(params_list)
            param_names, dims = zip(*((param["name"], param["dims"]) for param in params_list))
            constrained_param_names = sum((tuple(param["constrained_names"]) for param in params_list), ())
            return Model(model_name, program_code, data, param_names, constrained_param_names, dims, random_seed)

    try:
        return asyncio.run(go())
    except KeyboardInterrupt:
        return  # type: ignore
