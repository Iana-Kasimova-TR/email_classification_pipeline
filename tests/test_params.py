from unittest.mock import mock_open, patch

import pytest
import yaml

from src.utils.load_parameters import read_params


def test_read_params():
    # Mock data
    mock_data = """
    key1: value1
    key2: value2
    """
    expected_result = {
        'key1': 'value1',
        'key2': 'value2'
    }

    m = mock_open(read_data=mock_data)

    with patch("builtins.open", m):
        result = read_params("params.yaml")
        assert result == expected_result

def test_empty_file():
    m = mock_open(read_data="")

    with patch("builtins.open", m):
        result = read_params("params.yaml")
        assert result == None

# Optional: Test for invalid YAML
def test_invalid_yaml():
    mock_data = """
    key1: value1
    key2
    """

    m = mock_open(read_data=mock_data)

    with patch("builtins.open", m):
        with pytest.raises(yaml.YAMLError):
            read_params("params.yaml")