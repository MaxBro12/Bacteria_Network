class OsException(Exception):
    def __init__(self):
        self.txt = 'Неизвестная OS'
        super().__init__(self.txt)


class ConfigException(Exception):
    def __init__(self) -> None:
        self.txt = 'Config load Exception'
        super().__init__(self.txt)
