class Instruction:

    def __init__(self, opcode: str, args: list[str | int | float], arg_types: str = None):
        self.opcode: str = opcode
        self.args: list[str | int | float] = args
        self.arg_types: str = arg_types

    def __str__(self) -> None:
        return f"\n< Instruction(opcode = {self.opcode}, args = {self.args}, arg_types = {self.arg_types}) >\n"
    
    def __repr__(self) -> None:
        return self.__str__()