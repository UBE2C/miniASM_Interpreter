from Custom_errors import RegisterError

class Registers:
    def __init__(self, number: int = 31):
        self.registers: dict[str, int] = {f"rx{i}": 0 for i in range(number)}
        self.registers.update({"lr" : 0})  # Link register

    def __str__(self) -> None:
        return f"< register names: {self.registers.keys()} >\n< register values: {self.registers.values()} >"

    def __repr__(self) -> None:
        return self.__str__()

    def read(self, name: str) -> int | float:
        return self.registers.get(name)
        
    def write(self, name: str, value: int | float) -> None:
        if isinstance(value, (int, float)):
            self.registers[name] = value
        else:
            raise RegisterError(f"Register value must be int or float, register {name} got {type(value)}.")
        
    def names(self) -> set[str]:
        return set(self.registers.keys())
    


