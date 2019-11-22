import collections.abc
import json
import time
import typing

import requests

import httpstan.models
import httpstan.schemas
import httpstan.services.arguments as arguments
import httpstan.utils
import stan.common
import stan.fit

import google.protobuf.internal.decoder
import httpstan.callbacks_writer_pb2 as callbacks_writer_pb2
import numpy as np


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

        with stan.common.httpstan_server() as server:
            host, port = server.host, server.port
            stan_outputs = [[] for _ in range(num_chains)]
            payloads = []
            for chain in range(1, num_chains + 1):
                payload = {"function": "stan::services::sample::hmc_nuts_diag_e_adapt"}
                payload.update(kwargs)
                payload["chain"] = chain
                payload["data"] = self.data
                if self.random_seed is not None:
                    payload["random_seed"] = self.random_seed

                # fit needs to know num_samples, num_warmup, num_thin, save_warmup
                # progress bar needs to know some of these
                num_warmup = payload.get(
                    "num_warmup", arguments.lookup_default(arguments.Method["SAMPLE"], "num_warmup")
                )
                num_samples = payload.get(
                    "num_samples",
                    arguments.lookup_default(arguments.Method["SAMPLE"], "num_samples"),
                )
                num_thin = payload.get(
                    "num_thin", arguments.lookup_default(arguments.Method["SAMPLE"], "num_thin")
                )
                save_warmup = payload.get(
                    "save_warmup",
                    arguments.lookup_default(arguments.Method["SAMPLE"], "save_warmup"),
                )
                payloads.append(payload)

            # WISHLIST(AR): rewriting this function in Cython (in httpstan) might speed things up
            def extract_protobuf_messages(fit_bytes):
                varint_decoder = google.protobuf.internal.decoder._DecodeVarint32
                next_pos, pos = 0, 0
                while pos < len(fit_bytes):
                    msg = callbacks_writer_pb2.WriterMessage()
                    next_pos, pos = varint_decoder(fit_bytes, pos)
                    msg.ParseFromString(fit_bytes[pos : pos + next_pos])
                    yield msg
                    pos += next_pos

            fits_url = f"http://{host}:{port}/v1/{self.model_name}/fits"
            operations = []
            for payload in payloads:
                r = requests.post(fits_url, json=payload)
                if r.status_code != 201:
                    raise RuntimeError(r.json()["message"])
                assert r.status_code == 201, r.status_code
                operations.append(r.json())

            while not all(operation["done"] for operation in operations):
                for operation in operations:
                    operation_name = operation["name"]
                    operation.update(
                        requests.get(f"http://{host}:{port}/v1/{operation_name}").json()
                    )
                time.sleep(0.1)

            stan_outputs = []
            for operation in operations:
                fit_name = operation["result"].get("name")
                if fit_name is None:  # operation["result"] is an error
                    assert not str(operation["result"]["code"]).startswith("2"), operation
                    raise RuntimeError(operation["result"]["message"])
                r = requests.get(f"http://{host}:{port}/v1/{fit_name}", json=payload)
                if r.status_code != 200:
                    response_payload = r.json()
                    assert "message" in response_payload, response_payload
                    raise RuntimeError(response_payload["message"])
                stan_outputs.append(tuple(extract_protobuf_messages(r.content)))
            for stan_output in stan_outputs:
                assert isinstance(stan_output, tuple), stan_output
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

    with stan.common.httpstan_server() as server:
        host, port = server.host, server.port

        path, payload = "/v1/models", {"program_code": program_code}
        response = requests.post(f"http://{host}:{port}{path}", json=payload)
        if response.status_code != 201:
            response_payload = response.json()
            assert "message" in response_payload, response_payload
            message = response_payload["message"]
            raise RuntimeError(message)
        response_payload = response.json()
        model_name = response_payload["name"]

        path, payload = f"/v1/{model_name}/params", {"data": data}
        response = requests.post(f"http://{host}:{port}{path}", json=payload)
        response_payload = response.json()
        if response.status_code != 200:
            raise RuntimeError(response_payload["message"])
        assert response_payload.get("name") == model_name, response_payload
        params_list = response_payload["params"]
        assert len({param["name"] for param in params_list}) == len(params_list)

        param_names, dims = zip(*((param["name"], param["dims"]) for param in params_list))
        constrained_param_names = sum(
            (tuple(param["constrained_names"]) for param in params_list), ()
        )
    return Model(
        model_name, program_code, data, param_names, constrained_param_names, dims, random_seed
    )
