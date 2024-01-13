from typing import Any, cast

from toml import load, dump
from ..debug import create_log


def read_toml(way: str) -> dict[str, Any]:
    """Считываем файл по пути way формата .toml и возвращаем словарь"""
    return dict(load(way))


def write_toml(dictionary, way: str):
    """Записываем словарь dictionary в toml файл по пути way"""
    with open(way, 'w') as toml_file:
        dump(dictionary, toml_file)


def update_dict_to_type(dictionary: dict, type_to):
    """Обновляет словарь до выбранного типа"""
    return cast(type_to, dictionary)


def toml_type_check(typed_dict: dict, checking_dict: dict) -> bool:
    """Возращает True, если типизация и ключи dict1 соответсвуют dict2"""
    try:
        if len(typed_dict) != len(checking_dict):
            create_log(
                f'TOML len: {len(typed_dict)} != {len(checking_dict)}'
            )
            return False

        for key in typed_dict.keys():
            if type(checking_dict[key]) != type(typed_dict[key]):
                create_log(
                    f'TOML type: {key}: ' +
                    f'{type(checking_dict[key])} != {type(typed_dict[key])}'
                )
                return False

        return True
    except KeyError:
        create_log('TOML key error!')
        return False
