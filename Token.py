class Token:
    def __init__(self, type: str = None, value: str = None) -> None:
        self.type: str = type
        self.value: str = value
        

    def __str__(self) -> str:
        return f"\n< Token(type = {self.type}, value = {self.value}) >\n"
    
    def __repr__(self) -> str:
        return self.__str__()