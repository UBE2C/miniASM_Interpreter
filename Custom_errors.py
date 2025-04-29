class RegisterError(Exception):
    def __init__(self, message):
        super().__init__(message)

class vRAMError(Exception):
    def __init__(self, message):
        super().__init__(message)