from sys import argv
from core import create_log


def main(args: list):
    pass


if __name__ == '__main__':
    try:
        main(argv)
    except Exception as err:
        print(f'ERROR:\n{err}')
        create_log(err, 'crit')

