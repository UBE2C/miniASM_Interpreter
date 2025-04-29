from Custom_errors import vRAMError

class Memory:
    def __init__(self, size: int = 256) -> None:
        self.size = size
        self.vram: dict[int, int | float | str | bool] = {i : 0 for i in range(size)}

    def __str__(self) -> str:
        return f"< Virtual memory (vRAM) with {self.size} cells. >"

    def __repr__(self) -> str:
        return self.__str__()

    def view_vRAM(self) -> dict[int, Any]:
        return self.vram

    def view_address_lst(self) -> list[int]:
        return list(self.vram.keys())

    def check_free(self, address: int) -> bool:
        if self.vram.get(address) == 0:
            return True
        else:
            return False

    def list_free(self) -> list[int]:
        return_list: list[int] = []
        
        for key in self.vram:
            if self.vram.get(key) == 0:
                return_list.append(key)

        return return_list

    def list_occupied(self) -> dict[int, Any]:
        occupied_list: list[int] = []
        return_dict: dict[int, Any] = {}
        
        for key in self.vram:
            if self.vram.get(key) != 0:
                occupied_list.append(key)

        for key in occupied_list:
            return_dict[key] = self.vram.get(key)

        return return_dict

    def read(self, address: int) -> Any:
        if not isinstance(address, (int)):
            raise vRAMError(f"The given read address: {address} is not of type int.")

        elif isinstance(address, (int)) and address not in self.vram.keys():
            raise vRAMError(f"The given read address: {address} is out of bounds (size: {self.size})")
        
        else:
            return self.vram.get(address)

    def write(self, address: int, value: Any) -> None:
        if not isinstance(address, (int)):
            raise vRAMError(f"The given read address: {address} is not of type int.")

        elif isinstance(address, (int)) and address not in self.vram.keys():
            raise vRAMError(f"The given write address: {address} is out of bounds (size: {self.size})")
        
        else:
            self.vram[address] = value