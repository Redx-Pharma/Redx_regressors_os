import pytest
from redxregressors import utilities


def test_prepend_dictionary_keys():
    # Test case 1: Simple dictionary
    input_dict = {"a": 1, "b": 2}
    prepend_str = "prefix__"
    expected_output = {"prefix__a": 1, "prefix__b": 2}
    assert utilities.prepend_dictionary_keys(input_dict, prepend_str) == expected_output

    # Test case 2: Empty dictionary
    input_dict = {}
    prepend_str = "prefix__"
    expected_output = {}
    assert utilities.prepend_dictionary_keys(input_dict, prepend_str) == expected_output

    # Test case 3: Dictionary with different types of values
    input_dict = {"a": 1, "b": [1, 2, 3], "c": {"nested": "dict"}}
    prepend_str = "prefix__"
    expected_output = {
        "prefix__a": 1,
        "prefix__b": [1, 2, 3],
        "prefix__c": {"nested": "dict"},
    }
    assert utilities.prepend_dictionary_keys(input_dict, prepend_str) == expected_output

    # Test case 4: Dictionary with special characters in keys
    input_dict = {"a!": 1, "b@": 2}
    prepend_str = "prefix__"
    expected_output = {"prefix__a!": 1, "prefix__b@": 2}
    assert utilities.prepend_dictionary_keys(input_dict, prepend_str) == expected_output


if __name__ == "__main__":
    pytest.main()
