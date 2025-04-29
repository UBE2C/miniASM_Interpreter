class Instruction:

    def __init__(self, opcode: str, args: list[str | int | float]):
        self.opcode = opcode
        self.args = args

    def __str__(self) -> None:
        return f"Instruction(opcode = {self.opcode}, args = {self.args})"
    
    def __repr__(self) -> None:
        return self.__str__()