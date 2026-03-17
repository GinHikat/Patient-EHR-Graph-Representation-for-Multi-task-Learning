import pytest
from unittest.mock import MagicMock, patch
from services.graph_service import get_node_types_data

@patch("services.graph_service.get_db_driver")
def test_get_node_types_data(mock_get_driver):
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    mock_result = [{"label": "Person"}, {"label": "Movie"}]
    mock_session.run.return_value = mock_result
    
    mock_get_driver.return_value = mock_driver
    
    labels = get_node_types_data()
    assert labels == ["Person", "Movie"]
    mock_session.run.assert_called_once_with("CALL db.labels()")
