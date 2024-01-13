from ..debug import create_log

from os import rename, listdir, remove


def create_file(
    name: str,
    inner_text: str = ''
) -> bool:
    try:
        # ! Создание
        with open(name, 'w') as f:
            if inner_text != '':
                f.write(inner_text)
            f.close()
        return True
    except FileExistsError:
        create_log(f'Файл {name} уже существует!', 'error')
        return False
    except FileNotFoundError:
        create_log(f'Файл {name} не существует', 'error')
        return False


def save_file(name: str, inner: str) -> bool:
    try:
        with open(name, 'w', encoding='utf-8') as f:
            f.write(inner)
            return True
    except FileNotFoundError:
        create_log(f'Файл {name} не найден')
        return False


def load_file(name: str) -> str:
    try:
        create_log(f'Load: {name}')
        with open(name, 'r') as f:
            return f.read()
    except FileNotFoundError:
        create_log(f'Файл {name} не найден')
        return ''


def load_file_bytes(name: str) -> str:
    try:
        create_log(f'Load bytes: {name}')
        with open(name, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        create_log(f'Файл {name} не найден')
        return ''


def rename_file(last_name: str, new_name: str) -> bool:
    try:
        rename(last_name, new_name)
        return True
    except FileNotFoundError:
        create_log(f'Файл {last_name} не найден')
        return False
    except FileExistsError:
        create_log(
            f'Не возможно переименовать файл {last_name} в {new_name}! ' +
            'Файл с таким именем уже существует.', 'error'
        )
        return False


def delete_file(name: str) -> bool:
    try:
        remove(name)
        return True
    except FileNotFoundError:
        create_log(
            f'Не возможно удалить файл {name} файл не найден!', 'error'
        )
        return False


def get_files(where: str) -> list[str]:
    try:
        return listdir(where)
    except FileNotFoundError:
        create_log(
            f'Не возможно загрузить список файлов из директории {where}'
        )
        return []
