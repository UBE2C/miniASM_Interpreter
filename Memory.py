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





"""The pointer has to be updated to mark the position of each arra element, now that theere
is no unified offset!"""

class Pointer:
    def __init__(self, name: str, type: str, adrs: int, length: int, elements: list[int] = None):
        self.name: str = name
        self.type: str = type
        self.adrs: int = adrs
        self.length: int = length
        self.elements: list[int] = elements

    def __str__(self) -> str:
        return f"< Pointer for {self.name}, memory address: {self.adrs}, length {self.length} >"
    
    def __repr__(self) -> str:
        return self.__str__()


class Memory:
    def __init__(self, size: int = 262144) -> None:
        self.size = size
        self.vram: bytearray = bytearray(size)
        self.pointer_list: dict[str, Pointer] = {}
        self.occupied_addresses: list[int] = []

    def __str__(self) -> str:
        if len(self.occupied_addresses) == 0:
            return f"< Virtual memory (vRAM) with {self.size} cells ({self.size/1024} KB). >"
        else:
            return f"< Virtual memory (vRAM) with {self.size} cells ({self.size/1024} KB). >\n< The following addresses are occupied {self.occupied_addresses}>"

    def __repr__(self) -> str:
        return self.__str__()

    def view_vRAM(self) -> dict[int, bytearray]:
        return self.vram

    def check_occupied(self) -> None:
        output_lst: list[list[int]] = []
        if len(self.pointer_list) != 0:
            for key in self.pointer_list.keys():
                output_lst.append(list[self.pointer_list.get(key).adrs,  self.pointer_list.get(key).adrs + self.pointer_list.get(key).length])
        
        self.occupied_addresses = output_lst

    def view_section(self, pntr: str) -> str:
        section_start: int = self.pointer_list.get(pntr).adrs
        section_end: int = section_start + self.pointer_list.get(pntr).length
        return f"{self.vram[section_start:section_end]}"
    
    def int_to_byte(self, value: int) -> bytearray:
        if isinstance(value, (int)):
            if value >= (-2 ** 8) / 2 and value <= ((2 ** 8) / 2)-1: #8 bits, little-endian
                frm: str = "<b"
            
            elif value >= (-2 ** 16) / 2 and value <= ((2 ** 16) / 2)-1: #16 bits, little-endian
                frm: str = "<h"

            elif value >= (-2 ** 32) / 2 and value <= ((2 ** 32) / 2)-1: #32 bits, little-endian
                frm: str = "<i"

            elif value >= (-2 ** 64) / 2 and value <= ((2 ** 64) / 2)-1: #64 bits, little-endian
                frm: str = "<q"

            else:
                raise vRAMError(f"The supplied value: {value} exceeds the 64-bit limit.")

            return struct.pack(frm, value)

        else:
            raise vRAMError(f"Int to byte expected an integer argument but got {type(value)}")
    
    def float_to_byte(self, value: float) -> bytearray:
        if isinstance(value, (float)):
            if value >= (-2 ** 32) / 2 and value <= ((2 ** 32) / 2)-1: #32 bits, little-endian
                frm: str = "<f"

            elif value >= (-2 ** 64) / 2 and value <= ((2 ** 64) / 2)-1: #64 bits, little-endian
                frm: str = "<d"

            else:
                raise vRAMError(f"The supplied value: {value} exceeds the 64-bit limit.")

            return struct.pack(frm, value)

        else:
            raise vRAMError(f"Int to byte expected a float argument but got {type(value)}")

    def array_to_byte(self, obj: list[int | float], array_type: str, adrs: int) -> bytearray:
        byte_object: bytearray = bytearray()
        byte_buffer: bytearray = bytearray()
        element_positions: list[int] = []
        
        if not isinstance(obj, (list)) or len(obj) == 0:
            raise vRAMError(f"alloc.array_to_bytes: expected a non-empty list, got type: {type(obj)} with length {len(obj)}.")
        
        if array_type == "iarray" and not isinstance(obj[0], (int)):
            raise vRAMError(f"alloc.array_to_bytes: element type mismatch, element type {type(obj[0])} when int was expected.")

        if array_type == "farray" and not isinstance(obj[0], (float)):
            raise vRAMError(f"alloc.array_to_bytes: element type mismatch, element type {type(obj[0])} when float was expected.")

        if array_type not in {"iarray", "farray"}:
            raise vRAMError(f"alloc.array_to_bytes: unknown array type was supplied.")

        if isinstance(obj[0], (int)) and array_type == "iarray":
            expected_type: type = int  

        else:
            expected_type: type = float   

        for element in obj:
            if not isinstance(element, (expected_type)):
                raise vRAMError(f"alloc.array_to_bytes: mixed element type in array, an element is of {type(element)}, when {expected_type} was expected") 

        for element in obj:
            if isinstance(element, (int)):
                byte_object = self.int_to_byte(element)
                
            elif isinstance(element, (float)):
                byte_object = self.float_to_byte(element)

            len_byte_objet: int = len(byte_object)
            byte_buffer.extend(byte_object)
            element_positions.append(len(byte_buffer))

        return byte_buffer, element_positions
            

    def alloc(self, var_name: str, obj: int | float | str | list[int | float | str], type: str, adrs: int) -> Pointer:
        if adrs < 0 or adrs > self.size:
            raise vRAMError(f"The supplied memory address is out of bounds: address {adrs}, vRAM addresses: {0} to {len(self.vram)}")
        
        else:
            if type == "int":
                byte_obj: bytearray = self.int_to_byte(obj)
                offset: int = len(byte_obj)
                self.vram[adrs:adrs + offset] = byte_obj
                self.pointer_list[var_name] = Pointer(name = var_name, 
                                                type = type,
                                                adrs = adrs,
                                                length = offset)
            
            elif type == "float":
                byte_obj: bytearray = self.float_to_byte(obj)
                offset: int = len(byte_obj)
                self.vram[adrs:adrs + offset] = byte_obj
                self.pointer_list[var_name] = Pointer(name = var_name, 
                                                type = type,
                                                adrs = adrs,
                                                length = offset)
                
            elif type == "char":
                self.vram[adrs] = obj.encode(encoding = "utf-8")
                self.pointer_list[var_name] = Pointer(name = var_name, 
                                                type = type,
                                                adrs = adrs,
                                                length = 1)
                
            elif type == "iarray" or type == "farray": 
                buffer: bytearray = bytearray()
                element_positions: list[int] = []
                
                buffer, element_positions = self.array_to_byte(obj = obj, array_type = type, adrs = adrs)
                obj_len: int = len(buffer)

                if adrs + obj_len > len(self.vram):
                    raise vRAMError(f"The supplied array is out of bounds: array len: {obj_len}, vRAM len: {len(self.vram)}")
                
                else:
                    self.vram[adrs : adrs + obj_len] = buffer
                    self.pointer_list[var_name] = Pointer(name = var_name, 
                                                    type = type,
                                                    adrs = adrs,
                                                    length = obj_len,
                                                    elements = element_positions)

            self.check_occupied()
            

"""To do
wite a dealloc/free function to free memory regions
Check if I can do math operations with byte arrays
Convert the registers to bytearrays as needed/possible
Re-tuch the lexer, parse and executor with the additional sections and new opcodes
Implement bitwise operations
"""