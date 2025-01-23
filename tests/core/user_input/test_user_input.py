"""test_user_input.py"""
#import pytest
from unittest.mock import MagicMock
from hashin_shtrikman_mp.core.user_input.user_input import UserInput
from hashin_shtrikman_mp.core.user_input.material import Material
from hashin_shtrikman_mp.core.user_input.mixture import Mixture

def test_user_input_build_dict():
    mat_mock = MagicMock(spec=Material)
    mat_mock.custom_dict.return_value = {'mat_1': 'value_1'}
    
    mixture_mock = MagicMock(spec=Mixture)
    mixture_mock.custom_dict.return_value = {'mixture_1': 'value_2'}
    
    user_input = UserInput(materials=[mat_mock], mixtures=[mixture_mock])
    result = user_input.build_dict()
    assert result == {'mat_1': 'value_1', 'mixture_1': 'value_2'}

def test_user_input_items():
    user_input = UserInput(materials=[], mixtures=[])
    assert list(user_input.items()) == []

def test_user_input_keys():
    mat_mock = MagicMock(spec=Material)
    mat_mock.custom_dict.return_value = {'key1': 'value1'}
    
    user_input = UserInput(materials=[mat_mock], mixtures=[])
    assert list(user_input.keys()) == ['key1']

def test_user_input_values():
    mat_mock = MagicMock(spec=Material)
    mat_mock.custom_dict.return_value = {'key1': 'value1'}
    
    user_input = UserInput(materials=[mat_mock], mixtures=[])
    assert list(user_input.values()) == ['value1']

def test_user_input_len():
    mat_mock = MagicMock(spec=Material)
    mat_mock.custom_dict.return_value = {'key1': 'value1'}
    
    user_input = UserInput(materials=[mat_mock], mixtures=[])
    assert len(user_input) == 1

def test_user_input_getitem():
    mat_mock = MagicMock(spec=Material)
    mat_mock.custom_dict.return_value = {'key1': 'value1'}
    
    user_input = UserInput(materials=[mat_mock], mixtures=[])
    assert user_input['key1'] == 'value1'

def test_user_input_get():
    mat_mock = MagicMock(spec=Material)
    mat_mock.custom_dict.return_value = {'key1': 'value1'}
    
    user_input = UserInput(materials=[mat_mock], mixtures=[])
    assert user_input.get('key1') == 'value1'
    assert user_input.get('key2', 'default') == 'default'

def test_user_input_str_repr():
    mat_mock = MagicMock(spec=Material)
    mat_mock.custom_dict.return_value = {'key1': 'value1'}
    
    user_input = UserInput(materials=[mat_mock], mixtures=[])
    assert str(user_input) == "{'key1': 'value1'}"
    assert repr(user_input) == "{'key1': 'value1'}"
