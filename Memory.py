from Custom_errors import vRAMError
import struct

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


class Pointer:
    def __init__(self, name: str, type: str, adrs: int, length: int):
        self.name: str = name
        self.type: str = type
        self.adrs: int = adrs
        self.length: int = length

    def __str__(self) -> str:
        return f"< Pointer for {self.name}, memory address: {self.adrs}, length {self.length} >"
    
    def __repr__(self) -> str:
        return self.__str__()


class Memory:
    def __init__(self, size: int = 262144) -> None:
        self.size = size
        self.vram: bytearray = bytearray(size)
        self.pointer_list: dict[str, Pointer] = {}

    def __str__(self) -> str:
        return f"< Virtual memory (vRAM) with {self.size} cells ({self.size/1024} KB). >"

    def __repr__(self) -> str:
        return self.__str__()

    def view_vRAM(self) -> dict[int, bytearray]:
        return self.vram
    
    def int_to_byte(self, value: int) -> bytearray:
        if isinstance(value, (int)):
            return struct.pack(">q", value)
        else:
            raise vRAMError(f"Int to byte expected an integer argument but got {type(value)}")
    
    def float_to_byte(self, value: float) -> bytearray:
        if isinstance(value, (float)):
            return struct.pack(">d", value)
        else:
            raise vRAMError(f"Int to byte expected a float argument but got {type(value)}")
        
    def store_bytes(self, address: int, data: bytes) -> None:
        if address < 0 or address + len(data) > self.size:
            raise vRAMError(f"Memory access out of bounds: address {address}, length {len(data)}")
        
        for i, byte in enumerate(data):
            self.vram[address + i] = byte

    def alloc(self, var_name: str, obj: int | float | str | list[int | float | str], type: str, adrs: int) -> Pointer:
        if adrs < 0 or adrs > self.size:
            raise vRAMError(f"Memory access out of bounds: address {adrs}, length {len(self.vram)}")
        
        else:
            if type == "int":
                byte_obj: bytearray = self.int_to_byte(obj)
                offset: int = len(byte_obj)
                self.vram[adrs:adrs + offset] = byte_obj
                self.pointer_list[var_name] = Pointer(name = var_name, 
                                                type = type,
                                                adrs = adrs,
                                                length = offset)
            
            elif type == "double":
                self.vram[adrs] = self.float_to_byte(obj)
                self.pointer_list[var_name] = Pointer(name = var_name, 
                                                type = type,
                                                adrs = adrs,
                                                length = 1)
                
            elif type == "char":
                self.vram[adrs] = obj.encode(encoding = "utf-8")
                self.pointer_list[var_name] = Pointer(name = var_name, 
                                                type = type,
                                                adrs = adrs,
                                                length = 1)
                
            elif type == "iarray": 
                for i, integer in enumerate(obj):
                    byte_obj: bytearray = self.int_to_byte(integer)
                    self.vram[adrs + (i * 8)] = byte_obj
                
                obj_len: int = list[obj]
                self.pointer_list[var_name] = Pointer(name = var_name, 
                                                type = type,
                                                adrs = adrs,
                                                length = obj_len)