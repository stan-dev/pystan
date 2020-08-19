import asyncio
import collections.abc
import json
import re
import typing

import aiohttp
import google.protobuf.internal.decoder
import httpstan.callbacks_writer_pb2 as callbacks_writer_pb2
import httpstan.models
import httpstan.schemas
import httpstan.services.arguments as arguments
import httpstan.utils
import numpy as np
from clikit.io import ConsoleIO
from clikit.ui.components import ProgressBar

import stan.common
import stan.fit


def _make_json_serializable(data: dict) -> dict:
    """Convert `data` with numpy.ndarray-like values to JSON-serializable form.

    Returns a new dictionary.

    Arguments:
        data (dict): A Python dictionary or mapping providing the data for the
            model. Variable names are the keys and the values are their
            associated values. Default is an empty dictionary.

    Returns:
        dict: Copy of `data` dict with JSON-serializable values.
    """
    # no need for deep copy, we do not modify mutable items
    data = data.copy()
    for key, value in data.items():
        # first, see if the value is already JSON-serializable
        try:
            json.dumps(value)
        except TypeError:
            pass
        else:
            continue
        # numpy scalar
        if isinstance(value, np.ndarray) and value.ndim == 0:
            data[key] = np.asarray(value).tolist()
        # numpy.ndarray, pandas.Series, and anything similar
        elif isinstance(value, collections.abc.Collection):
            data[key] = np.asarray(value).tolist()
        else:
            raise TypeError(f"Value associated with variable `{key}` is not JSON serializable.")
    return data


class Model:
    """Stores data associated with and proxies calls to a Stan model.

    Returned by ``build``. Users will not instantiate this class directly.

    """

    def __init__(
        self,
        model_name: str,
        program_code: str,
        data: dict,
        param_names: typing.Tuple[str],
        constrained_param_names: typing.Tuple[str],
        dims: typing.Tuple[typing.Tuple[int]],
        random_seed: typing.Optional[int],
    ) -> None:
        if model_name != httpstan.models.calculate_model_name(program_code):
            raise ValueError("`model_name` does not match `program_code`.")
        self.model_name = model_name
        self.program_code = program_code
        self.data = data or {}
        self.param_names = param_names
        self.constrained_param_names = constrained_param_names
        self.dims = dims
        self.random_seed = random_seed

    def sample(self, **kwargs):
        """Draw samples from the model.

        Parameters in ``kwargs`` will be passed to the default sample function in
        stan::services. Parameter names are identical to those used in CmdStan.
        See the CmdStan documentation for parameter descriptions and default
        values.

        Returns:
            Fit: instance of Fit allowing access to draws.

        """
        assert isinstance(self.data, dict)
        assert "chain" not in kwargs, "`chain` id is set automatically."
        assert "data" not in kwargs, "`data` is set in `build`."
        assert "random_seed" not in kwargs, "`random_seed` is set in `build`."
        num_chains = kwargs.pop("num_chains", 1)

        init = kwargs.pop("init", [dict() for _ in range(num_chains)])
        if len(init) != num_chains:
            raise ValueError("Initial values must be provided for each chain.")

        payloads = []
        for chain in range(1, num_chains + 1):
            payload = {"function": "stan::services::sample::hmc_nuts_diag_e_adapt"}
            payload.update(kwargs)
            payload["chain"] = chain
            payload["data"] = self.data
            payload["init"] = init.pop(0)
            if self.random_seed is not None:
                payload["random_seed"] = self.random_seed

            # fit needs to know num_samples, num_warmup, num_thin, save_warmup
            # progress bar needs to know some of these
            num_warmup = payload.get("num_warmup", arguments.lookup_default(arguments.Method["SAMPLE"], "num_warmup"))
            num_samples = payload.get(
                "num_samples", arguments.lookup_default(arguments.Method["SAMPLE"], "num_samples"),
            )
            num_thin = payload.get("num_thin", arguments.lookup_default(arguments.Method["SAMPLE"], "num_thin"))
            save_warmup = payload.get(
                "save_warmup", arguments.lookup_default(arguments.Method["SAMPLE"], "save_warmup"),
            )
            payloads.append(payload)

        def extract_protobuf_messages(fit_bytes):
            varint_decoder = google.protobuf.internal.decoder._DecodeVarint32
            next_pos, pos = 0, 0
            while pos < len(fit_bytes):
                msg = callbacks_writer_pb2.WriterMessage()
                next_pos, pos = varint_decoder(fit_bytes, pos)
                msg.ParseFromString(fit_bytes[pos : pos + next_pos])
                yield msg
                pos += next_pos

        async def go():
            io = ConsoleIO()
            progress_bar = ProgressBar(io)
            progress_bar.set_format("very_verbose")
            progress_bar.set_message("Sampling...")

            current_and_max_iterations_re = re.compile(r"Iteration:\s+(\d+)\s+/\s+(\d+)")
            async with stan.common.httpstan_server() as (host, port):
                stan_outputs = [[] for _ in range(num_chains)]

                fits_url = f"http://{host}:{port}/v1/{self.model_name}/fits"
                operations = []
                for payload in payloads:
                    async with aiohttp.request("POST", fits_url, json=payload) as resp:
                        if resp.status == 422:
                            raise ValueError(str(await resp.json()))
                        elif resp.status != 201:
                            raise RuntimeError((await resp.json())["message"])
                        assert resp.status == 201
                        operations.append(await resp.json())

                # poll to get progress for each chain until all chains finished
                current_iterations = {}
                while not all(operation["done"] for operation in operations):
                    for operation in operations:
                        if operation["done"]:
                            continue
                        operation_name = operation["name"]
                        async with aiohttp.request("GET", f"http://{host}:{port}/v1/{operation_name}") as resp:
                            operation.update(await resp.json())
                            progress_message = operation["metadata"].get("progress")
                            if not progress_message:
                                continue
                            iteration, iteration_max = map(
                                int, current_and_max_iterations_re.findall(progress_message).pop(0)
                            )
                            if not progress_bar.get_max_steps():  # i.e., has not started
                                progress_bar.start(max=iteration_max * num_chains)
                            current_iterations[operation["name"]] = iteration
                            progress_bar.set_progress(sum(current_iterations.values()))
                    await asyncio.sleep(0.01)
                progress_bar.set_message("Sampling finished.")
                progress_bar.finish()

                stan_outputs = []
                for operation in operations:
                    fit_name = operation["result"].get("name")
                    if fit_name is None:  # operation["result"] is an error
                        assert not str(operation["result"]["code"]).startswith("2"), operation
                        raise RuntimeError(operation["result"]["message"])
                    async with aiohttp.request("GET", f"http://{host}:{port}/v1/{fit_name}") as resp:
                        if resp.status != 200:
                            raise RuntimeError((await resp.json())["message"])
                        stan_outputs.append(tuple(extract_protobuf_messages(await resp.read())))

                def is_nonempty_logger_message(msg):
                    return (
                        msg.topic == callbacks_writer_pb2.WriterMessage.Topic.LOGGER
                        and msg.feature[0].string_list.value[0].strip() != "info:"
                    )

                def is_iteration_or_elapsed_time_logger_message(msg):
                    # Assumes `msg` is a message with topic `LOGGER`.
                    text = msg.feature[0].string_list.value[0]
                    return (
                        text.startswith("info:Iteration:")
                        or text.startswith("info: Elapsed Time:")
                        # this detects lines following "Elapsed Time:", part of a multi-line Stan message
                        or text.startswith("info:" + " " * 15)
                    )

                logger_messages = []
                for stan_output in stan_outputs:
                    assert isinstance(stan_output, tuple), stan_output
                    logger_messages.extend(filter(is_nonempty_logger_message, stan_output))

                non_standard_logger_messages = list(
                    filter(lambda msg: not is_iteration_or_elapsed_time_logger_message(msg), logger_messages)
                )
                if non_standard_logger_messages:
                    io.error("\n<info>Messages received during sampling:</info>\n")
                    for msg in non_standard_logger_messages:
                        text = msg.feature[0].string_list.value[0].replace("info:", "  ")
                        io.error(f"<info>{text}</info>\n")
            return stan.fit.Fit(
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

        return asyncio.run(go())


def build(program_code, data=None, random_seed=None):
    """Build (compile) a Stan program.

    Arguments:
        program_code (str): Stan program code describing a Stan model.
        data (dict): A Python dictionary or mapping providing the data for the
            model. Variable names are the keys and the values are their
            associated values. Default is an empty dictionary.
        random_seed (int): Random seed, a positive integer for random number
            generation. Used to make sure that results can be reproduced.

    Returns:
        Model: an instance of Model

    Notes:
        C++ reserved words and Stan reserved words may not be used for
        variable names; see the Stan User's Guide for a complete list.

    """
    if data is None:
        data = {}
    # _make_json_serializable returns a new dict, original `data` unchanged
    data = _make_json_serializable(data)
    assert all(not isinstance(value, np.ndarray) for value in data.values())

    async def go():
        io = ConsoleIO()
        io.error("<info>Building...</info>")
        async with stan.common.httpstan_server() as (host, port):
            # Check to see if model is in cache.
            model_name = httpstan.models.calculate_model_name(program_code)
            path, payload = f"/v1/{model_name}/params", {"data": data}
            async with aiohttp.request("POST", f"http://{host}:{port}{path}", json=payload) as resp:
                model_in_cache = resp.status != 404
            io.error_line(" Found model in cache." if model_in_cache else " This may take some time.")
            # Note: during compilation `httpstan` redirects stderr to /dev/null, making `print` impossible.
            path, payload = "/v1/models", {"program_code": program_code}
            async with aiohttp.request("POST", f"http://{host}:{port}{path}", json=payload) as resp:
                if resp.status != 201:
                    raise RuntimeError((await resp.json())["message"])
                assert model_name == (await resp.json())["name"]

            path, payload = f"/v1/{model_name}/params", {"data": data}
            async with aiohttp.request("POST", f"http://{host}:{port}{path}", json=payload) as resp:
                if resp.status != 200:
                    raise RuntimeError((await resp.json())["message"])
                params_list = (await resp.json())["params"]
            assert len({param["name"] for param in params_list}) == len(params_list)
            param_names, dims = zip(*((param["name"], param["dims"]) for param in params_list))
            constrained_param_names = sum((tuple(param["constrained_names"]) for param in params_list), ())
            return Model(model_name, program_code, data, param_names, constrained_param_names, dims, random_seed)

    return asyncio.run(go())
