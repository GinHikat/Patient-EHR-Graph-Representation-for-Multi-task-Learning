import pytest
from fastapi.testclient import TestClient
from main import app
from unittest.mock import patch

client = TestClient(app)

@patch("api.routes.get_node_types_data")
def test_get_node_types_system(mock_get_types):
    mock_get_types.return_value = ["TestLabel1", "TestLabel2"]
    response = client.get("/api/node_types")
    assert response.status_code == 200
    assert response.json() == {"node_types": ["TestLabel1", "TestLabel2"]}

@patch("api.routes.get_graph_data")
def test_get_graph_system(mock_get_graph):
    mock_get_graph.return_value = {
        "nodes": [{"id": "1", "labels": ["A"], "properties": {}}],
        "links": []
    }
    response = client.get("/api/graph?limit=10")
    assert response.status_code == 200
    assert response.json() == {
        "nodes": [{"id": "1", "labels": ["A"], "properties": {}}],
        "links": []
    }
