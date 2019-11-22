import requests

import stan


def test_httpstan_health():
    with stan.common.httpstan_server() as server:
        host, port = server.host, server.port
        response = requests.get(f"http://{host}:{port}/v1/health")
        assert response.status_code == 200
