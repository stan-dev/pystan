"""Test model parameter shapes."""
import pytest

import stan

program_code = """
    data {
      int K;
      int L;
      int M;
      int N;
      int O;
      int P;
      int Q;
      int R;
      int S;
    }
    parameters {
      real a[K];
      real B[L, M];
      vector[N] c;
      matrix[O, P] D;
      matrix[R, S] E[Q];
    }
    model {
      for (k in 1:K) {
        a[k] ~ std_normal();
      }

      for (l in 1:L) {
        for (m in 1:M) {
          B[l, m] ~ std_normal();
        }
      }

      for (n in 1:N) {
        c[n] ~ std_normal();
      }

      for (o in 1:O) {
        for (p in 1:P) {
          D[o, p] ~ std_normal();
        }
      }

      for (q in 1:Q) {
        for (r in 1:R) {
          for (s in 1:S) {
            E[q, r, s] ~ std_normal();
          }
        }
      }

    }
"""
num_samples = 100
num_chains = 3

dims = {
    "a": ("K",),
    "B": ("L", "M"),
    "c": ("N",),
    "D": ("O", "P"),
    "E": ("Q", "R", "S"),
}


def get_posterior(data):
    return stan.build(program_code, data=data)


def get_fit(data):
    posterior = get_posterior(data)
    return posterior.sample(num_samples=num_samples, num_chains=num_chains)


def get_data(zero_dims):
    data = {
        "K": 2,
        "L": 3,
        "M": 2,
        "N": 2,
        "O": 3,
        "P": 2,
        "Q": 4,
        "R": 3,
        "S": 2,
    }
    for zero_dim in zero_dims:
        assert zero_dim in data
        data[zero_dim] = 0
    return data


@pytest.mark.parametrize(
    "zero_dims",
    ["K", "L", "M", "LM", "N", "O", "P", "OP", "Q", "R", "S", "QR", "QS", "RS", "QRS", "LMNOPQRS"],
)
def test_fit_empty_array_shape(zero_dims):
    """
    Make sure shapes are correct.
    """
    data = get_data(zero_dims)
    fit = get_fit(data)
    for parameter, dim in dims.items():
        base_shape = tuple(map(data.get, dim))
        assert fit[parameter].shape == base_shape + (num_samples * num_chains,)
        assert fit.get_samples(parameter, flatten_chains=False).shape == base_shape + (num_samples, num_chains)
