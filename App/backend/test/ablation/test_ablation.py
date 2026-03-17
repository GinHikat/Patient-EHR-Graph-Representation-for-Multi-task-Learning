import pytest
from services.graph_service import get_graph_data
from unittest.mock import patch, MagicMock

# Ablation test: Test the system's robustness when link processing is omitted/fails
@patch("services.graph_service.get_db_driver")
def test_get_graph_data_ablation_no_links(mock_get_driver):
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    
    class MockNode:
        def __init__(self, id, labels):
            self.element_id = id
            self.labels = labels
            
        def __iter__(self):
            return iter([])
            
    # Mocking a result where 'm' and 'n' are present but 'r' is None
    # Simulate an ablation where edges are missing
    mock_session.run.return_value = [
        {"n": MockNode("1", ["A"]), "m": MockNode("2", ["B"]), "r": None}
    ]
    mock_get_driver.return_value = mock_driver
    
    data = get_graph_data(limit=10)
    
    assert len(data["nodes"]) == 2
    assert "1" in [node["id"] for node in data["nodes"]]
    assert "2" in [node["id"] for node in data["nodes"]]
    assert len(data["links"]) == 0
