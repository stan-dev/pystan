import socket

import requests
import pytest

import stan


def test_httpstan_port_conflict():
    s = socket.socket()
    try:
        s.bind(("", 8080))
        with pytest.raises(OSError) as excinfo:
            with stan.common.httpstan_server() as server:
                host, port = server.host, server.port
                response = requests.get(f"http://{host}:{port}/v1/health")
                assert response.status_code == 200
            assert "address already in use" in str(excinfo.value)
    finally:
        s.close()
