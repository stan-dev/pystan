import concurrent.futures
import json
import typing

import requests
import tqdm

import httpstan.models
import httpstan.services.arguments as arguments
import pystan.common
import pystan.fit


class Model:
    """Stores data associated with and proxies calls to a Stan model.

    Returned by `compile`. Users will not instantiate this class directly.

    """

    def __init__(
        self,
        model_id: str,
        program_code: str,
        data: dict,
        param_names: typing.Tuple[str],
        constrained_param_names: typing.Tuple[str],
        dims: typing.Tuple[typing.Tuple[int]],
        random_seed: int,
    ) -> None:
        if model_id != httpstan.models.calculate_model_id(program_code):
            raise ValueError("`model_id` does not match `program_code`.")
        self.model_id = model_id
        self.program_code = program_code
        self.data = data or {}
        self.param_names = param_names
        self.constrained_param_names = constrained_param_names
        self.dims = dims
        self.random_seed = random_seed

    def sample(self, **kwargs):
        """Draw samples from the model.

        Parameters in `kwargs` will be passed to the default sample function in
        stan::services. Parameter names are identical to those used in CmdStan.
        See the CmdStan documentation for parameter descriptions and default
        values.

        Returns:
            Fit: instance of Fit allowing access to draws.

        """
        assert isinstance(self.data, dict)
        assert "chain" not in kwargs, "`chain` id is set automatically."
        assert "data" not in kwargs, "`data` is set in `compile`."
        assert "random_seed" not in kwargs, "`random_seed` is set in `compile`."
        num_chains = kwargs.pop("num_chains", 1)

        def go(num_chains):
            with pystan.common.httpstan_server() as server:
                host, port = server.host, server.port
                path = f"/v1/models/{self.model_id}/actions"
                stan_outputs = [[] for _ in range(num_chains)]
                payloads = []
                for chain in range(1, num_chains + 1):
                    payload = {"type": "stan::services::sample::hmc_nuts_diag_e"}
                    payload.update(kwargs)
                    payload["chain"] = chain
                    payload["data"] = self.data

                    # fit needs to know num_samples, num_warmup, num_thin, save_warmup
                    # progress bar needs to know some of these
                    num_warmup = payload.get(
                        "num_warmup",
                        arguments.lookup_default(arguments.Method["SAMPLE"], "num_warmup"),
                    )
                    num_samples = payload.get(
                        "num_samples",
                        arguments.lookup_default(arguments.Method["SAMPLE"], "num_samples"),
                    )
                    num_thin = payload.get(
                        "num_thin",
                        arguments.lookup_default(arguments.Method["SAMPLE"], "num_thin"),
                    )
                    save_warmup = payload.get(
                        "save_warmup",
                        arguments.lookup_default(arguments.Method["SAMPLE"], "save_warmup"),
                    )
                    pbar_total = num_samples + num_warmup * int(save_warmup)
                    payloads.append(payload)

                def gather_draws(payload, pbar=None):
                    stan_output = []
                    r = requests.post(f"http://{host}:{port}{path}", json=payload, stream=True)
                    for line in r.iter_lines():
                        payload_response = json.loads(line)
                        stan_output.append(payload_response)
                        if payload_response["topic"] == "SAMPLE":
                            if pbar:
                                pbar.update()
                    return stan_output

                pbars = [
                    tqdm.tqdm(total=pbar_total, position=i, desc=f"Chain {i + 1}")
                    for i in range(num_chains)
                ]
                with concurrent.futures.ThreadPoolExecutor(num_chains) as executor:
                    futures = [
                        executor.submit(gather_draws, payload, pbar)
                        for payload, pbar in zip(payloads, pbars)
                    ]
                    stan_outputs = [fut.result() for fut in futures]
                for pbar in pbars:
                    pbar.close()

                for stan_output in stan_outputs:
                    assert isinstance(stan_output, list), stan_output
            return pystan.fit.Fit(
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

        return go(num_chains)


def compile(program_code, data=None, random_seed=None):
    """Compile a Stan program.

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
    with pystan.common.httpstan_server() as server:
        host, port = server.host, server.port

        path, payload = "/v1/models", {"program_code": program_code}
        response_payload = requests.post(f"http://{host}:{port}{path}", data=payload).json()
        model_id = response_payload["id"]

        path, payload = f"/v1/models/{model_id}/params", {"data": data}
        response_payload = requests.post(f"http://{host}:{port}{path}", json=payload).json()
        assert response_payload.get("id") == model_id, response_payload
        params_list = response_payload["params"]
        assert len({param["name"] for param in params_list}) == len(params_list)

        param_names, dims = zip(*((param["name"], param["dims"]) for param in params_list))
        constrained_param_names = sum(
            (tuple(param["constrained_names"]) for param in params_list), ()
        )
    return Model(
        model_id, program_code, data, param_names, constrained_param_names, dims, random_seed
    )
