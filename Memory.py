from Custom_errors import vRAMError
import struct
import warnings
from Registers import Register
from Registers import RegisterSupervisor

class Pointer:
    def __init__(self, var_name: str, var_type: str, adrs: int, length: int, element_offsets: list[int] | None = None):
        self.var_name: str = var_name
        self.var_type: str = var_type
        self.adrs: int = adrs
        self.length: int = length
        
        if element_offsets == None:
            self.element_offsets: list[int] = []
        else:
            self.element_offsets: list[int] = element_offsets
        self.element_indexes: list[list[int]] = []
        self.write_window_start: int = self.adrs


    def __str__(self) -> str:
        return f"< Pointer for {self.var_name} of type {self.var_type}, memory address: {self.adrs}, length {self.length}, element offsets {self.element_offsets}, element indexes {self.element_indexes} >"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def indexer(self, offset_lst: list[int]) -> list[list[int]]:
        indexes: list[list[int]] = []
        previous: int = 0
        for offset in offset_lst:
            indexes.append([previous, previous + offset])
            previous = previous + offset

        self.element_indexes = indexes
        return indexes


class Memory:
    def __init__(self, size: int = 1048576) -> None:
        self.size = size
        self.vram: bytearray = bytearray(size)
        self.pointer_list: dict[str, Pointer] = {}
        self.occupied_addresses: list[int] = []

    def __str__(self) -> str:
        if len(self.occupied_addresses) == 0:
            return f"< Virtual memory (vRAM) with {self.size} cells ({self.size/1048576} MB). >"
        else:
            return f"< Virtual memory (vRAM) with {self.size} cells ({self.size/1048576} MB). >\n< The following addresses are occupied {self.occupied_addresses}>"

    def __repr__(self) -> str:
        return self.__str__()

    def view_vRAM(self) -> dict[int, bytearray]:
        return self.vram

    def check_occupied(self) -> None:
        output_lst: list[list[int]] = []
        if len(self.pointer_list) != 0:
            for key in self.pointer_list.keys():
                output_lst.append([self.pointer_list.get(key).adrs,  self.pointer_list.get(key).adrs + self.pointer_list.get(key).length])
        
        self.occupied_addresses = output_lst

    def view_section(self, pntr: str) -> str:
        section_start: int = self.pointer_list.get(pntr).adrs
        section_end: int = section_start + self.pointer_list.get(pntr).length
        
        return f"{self.vram[section_start:section_end]}"
    
    def numeric_to_byte(self, var: int | float, var_type: str) -> bytearray:
        if var_type == "int" and not isinstance(var, (int)):
            raise vRAMError(f"numeric_to_byte: expected an int based on {var_type}, but {type(var)} was supplied.")
        
        if var_type == "float" and not isinstance(var, (float)):
            raise vRAMError(f"numeric_to_byte: expected a float based on {var_type}, but {type(var)} was supplied.")

        if var_type not in {"int", "float"}:
            raise vRAMError(f"numeric_to_byte: unknown variable type was supplied.")
        
        if var_type == "int" and isinstance(var, (int)):
            if var >= (-2 ** 8) / 2 and var <= ((2 ** 8) / 2)-1: #8 bits, little-endian
                frm: str = "<b"
            
            elif var >= (-2 ** 16) / 2 and var <= ((2 ** 16) / 2)-1: #16 bits, little-endian
                frm: str = "<h"

            elif var >= (-2 ** 32) / 2 and var <= ((2 ** 32) / 2)-1: #32 bits, little-endian
                frm: str = "<i"

            elif var >= (-2 ** 64) / 2 and var <= ((2 ** 64) / 2)-1: #64 bits, little-endian
                frm: str = "<q"

            else:
                raise vRAMError(f"numeric_to_byte: The supplied variable: {var} exceeds the 64-bit limit.")

            return struct.pack(frm, var)
        
        elif var_type == "float" and isinstance(var, (float)):
            if var >= (-2 ** 32) / 2 and var <= ((2 ** 32) / 2)-1: #32 bits, little-endian
                frm: str = "<f"

            elif var >= (-2 ** 64) / 2 and var <= ((2 ** 64) / 2)-1: #64 bits, little-endian
                frm: str = "<d"

            else:
                raise vRAMError(f"numeric_to_byte: The supplied value: {var} exceeds the 64-bit limit.")

            return struct.pack(frm, var)

    def array_to_byte(self, obj: list[int | float], array_type: str) -> tuple[bytearray, list[int]]:
        byte_object: bytearray = bytearray()
        byte_buffer: bytearray = bytearray()
        element_offsets: list[int] = []
        
        if not isinstance(obj, (list)) or len(obj) == 0:
            raise vRAMError(f"array_to_bytes: expected a non-empty list, got type: {type(obj)} with length {len(obj)}.")
        
        if array_type == "iarray" and not isinstance(obj[0], (int)):
            raise vRAMError(f"array_to_bytes: element type mismatch, element type {type(obj[0])} when int was expected.")

        if array_type == "farray" and not isinstance(obj[0], (float)):
            raise vRAMError(f"array_to_bytes: element type mismatch, element type {type(obj[0])} when float was expected.")

        if array_type not in {"iarray", "farray"}:
            raise vRAMError(f"array_to_bytes: unknown array type was supplied.")

        if isinstance(obj[0], (int)) and array_type == "iarray":
            expected_type: type = int  
            var_type: str = "int"

        else:
            expected_type: type = float
            var_type: str = "float"

        for element in obj:
            if not isinstance(element, (expected_type)):
                raise vRAMError(f"array_to_bytes: mixed element type in array, an element is of {type(element)}, when {expected_type} was expected") 

        for element in obj:
            if isinstance(element, (int)):
                byte_object = self.numeric_to_byte(element, var_type)
                
            elif isinstance(element, (float)):
                byte_object = self.numeric_to_byte(element, var_type)

            byte_buffer.extend(byte_object)
            element_offsets.append(len(byte_buffer)) #element_offsets are buffer offsets

        return byte_buffer, element_offsets
            

    def store(self, var_name: str, obj: int | float | str | list[int | float | str], obj_type: str, adrs: int) -> Pointer:
        if adrs < 0 or adrs > self.size:
            raise vRAMError(f"store: The supplied memory address is out of bounds: address {adrs}, vRAM addresses: {0} to {len(self.vram)}")
        
        else:
            if obj_type == "iarray":
                element_type: str = "int"
        
            elif obj_type == "farray":
                element_type: str = "float"

            elif obj_type == "char":
                element_type: str = "char"
            
            elif obj_type == "string":
                element_type: str = "string"

            else:
                raise vRAMError(f"store: the supplied array contains elements with unsupported types: {type(obj[0])}")

            if obj_type == "int" or obj_type == "float":
                buffer: bytearray = self.numeric_to_byte(obj, element_type)
                obj_len: int = len(buffer)
                
                self.vram[adrs:adrs + obj_len] = buffer
                self.pointer_list[var_name] = Pointer(var_name = var_name, 
                                                var_type = obj_type,
                                                adrs = adrs,
                                                length = obj_len)
                
            elif obj_type == "iarray" or obj_type == "farray": 
                buffer: bytearray = bytearray()
                element_positions: list[int] = []
                
                buffer, element_positions = self.array_to_byte(obj = obj, array_type = obj_type)
                obj_len: int = len(buffer)

                if adrs + obj_len > len(self.vram):
                    raise vRAMError(f"store: The supplied array is out of bounds: array len: {obj_len}, vRAM len: {len(self.vram)}")
                
                else:
                    self.vram[adrs : adrs + obj_len] = buffer
                    self.pointer_list[var_name] = Pointer(var_name = var_name, 
                                                    var_type = obj_type,
                                                    adrs = adrs,
                                                    length = obj_len,
                                                    element_offsets = element_positions)
                    
            elif obj_type == "char" or obj_type == "string":
                buffer: bytearray = obj.encode(encoding = "iso-8859-2")
                obj_len: int = len(buffer)

                if adrs + obj_len > len(self.vram):
                    raise vRAMError(f"stor: The supplied char or string is out of bounds: array len: {obj_len}, vRAM len: {len(self.vram)}")
                
                else:
                    self.vram[adrs : adrs + obj_len] = buffer
                    self.pointer_list[var_name] = Pointer(var_name = var_name, 
                                                    var_type = obj_type,
                                                    adrs = adrs,
                                                    length = obj_len)
                
            
            ptr: Pointer = self.pointer_list[var_name]

            self.check_occupied()
            return ptr
            
    def reserve_over_limit(self, size: int, block_name: str) -> Pointer:
        if size <= 0:
            raise vRAMError(f"reserve_extra: the size of the extra block must be positive and larger than 0 bytes, {size} were provided.")
        
        if block_name in self.pointer_list.keys():
            raise vRAMError(f"reserve_extra: block '{block_name}' already exists")
        
        original_size: int = len(self.vram)
        self.vram.extend(bytearray(size))
        self.pointer_list.update({block_name : Pointer(var_name = block_name, 
                                                var_type = "reserved",
                                                adrs = original_size,
                                                length = size)})
        self.size = len(self.vram)
        self.check_occupied()
        
        ptr: Pointer = self.pointer_list.get(block_name)

        return ptr

    def reserve(self, size: int, adrs: int, block_name: str, overwrite: bool = False) -> Pointer:
        if size <= 0:
            raise vRAMError(f"reserve: the size of the reserved block must be positive and larger than 0 bytes, {size} were provided.")

        if block_name in self.pointer_list.keys():
            raise vRAMError(f"reserve: the block {block_name} is already reserved or occupied. Free the occupied block first before a new reserve.")

        if overwrite == False:
            if adrs in self.occupied_addresses:
                raise vRAMError(f"reserve: the given address {adrs} is already reserved or occupied. Free the occupied block first before a new reserve.")

            for block in self.occupied_addresses:
                block_start: int = block[0]
                block_end: int = block[1]
                
                if block[start] < (adrs + size) and block_end >= adrs:
                    raise vRAMError(f"reserve: the given memory block overlaps with an already occupied block {block}. Free the occupied block first or shift the new block.")

        self.pointer_list.update({block_name : Pointer(var_name = block_name,
                                                       var_type = "reserved",
                                                       adrs = adrs,
                                                       length = size)})

        self.check_occupied()
        ptr: Pointer = self.pointer_list.get(block_name)

        return ptr

    def dynamic_store(self, var_name: str, source: Register, obj_type: str) -> Pointer:
        if var_name not in self.pointer_list.keys() and self.pointer_list[var_name].var_type != "reserved":
            raise vRAMError(f"dynamic_store: the variable {var_name} cannot be allocated as no memory was reserved for it.")

        block_start: int = self.pointer_list.get(var_name).adrs
        block_end: int = self.pointer_list.get(var_name).adrs + self.pointer_list.get(var_name).length
        
        transfer_buffer: bytearray = bytearray()
        offset: int = 0
        current_write_window: int = self.pointer_list.get(var_name).write_window_start

        
        transfer_buffer = source.read_bytes()
        offset = len(transfer_buffer)

        if current_write_window + offset > block_end:
            raise vRAMError(f"dynamic store: the next current element form register {source.name} would exceed the length capacity of the reserved memory block.")
            
        self.vram[current_write_window : current_write_window + offset] = transfer_buffer
        self.pointer_list.get(var_name).write_window_start = current_write_window + offset
        self.pointer_list.get(var_name).element_offsets.append(offset)
        if obj_type == "int":
            self.pointer_list.get(var_name).var_type = "iarray"
        
        elif obj_type == "float":
            self.pointer_list.get(var_name).var_type = "farray"

        elif obj_type in {"char", "string"}:
            self.pointer_list.get(var_name).var_type = "string"

        self.pointer_list.get(var_name).indexer(self.pointer_list.get(var_name).element_offsets)

        ptr: Pointer = self.pointer_list.get(var_name)
        self.check_occupied()

        return ptr

    def write_reserved(self, var_name: str, obj: int | float | str | list[int | float | str], obj_type: str) -> Pointer:
        if var_name in self.pointer_list.keys() and self.pointer_list[var_name].var_type == "reserved":
            if obj_type in ("int", "float"):
                buffer: bytearray = self.numeric_to_byte(obj)
                block_start: int = self.pointer_list[var_name].adrs
                block_end: int = block_start + len(buffer)

                if len(buffer) > self.pointer_list[var_name].length:
                    raise vRAMError("data length exceeds reserved block size")
                
                else:
                    self.vram[block_start : block_end] = buffer
                    self.pointer_list[var_name].var_type = obj_type

            elif len(obj) > 0 and obj_type in ("iarray", "farray"):
                buffer: bytearray = bytearray()
                new_offsets: list[int] = []
                buffer, new_offsets = self.array_to_byte(obj = obj, array_type = obj_type)
                block_start: int = self.pointer_list[var_name].adrs
                block_end: int = block_start + len(buffer)

                if len(buffer) > self.pointer_list[var_name].length:
                    raise vRAMError("data length exceeds reserved block size")
                
                else:
                    self.vram[block_start : block_end] = buffer
                    self.pointer_list[var_name].var_type = obj_type
                    self.pointer_list[var_name].element_offsets = new_offsets
                    self.pointer_list[var_name].element_indexes = self.pointer_list[var_name].indexer(new_offsets)


            elif len(obj) > 0 and obj_type in ("char", "string"):
                buffer: bytearray = obj.encode(encoding = "iso-8859-2")
                block_start: int = self.pointer_list[var_name].adrs
                block_end: int = block_start + len(buffer)

                if len(buffer) > self.pointer_list[var_name].length:
                    raise vRAMError("data length exceeds reserved block size")
                
                else:
                    self.vram[block_start : block_end] = buffer
                    self.pointer_list[var_name].var_type = obj_type

            else:
                raise vRAMError(f"write_reserved: unknown variable type {obj_type} or empty variable {len(obj)}.")

        else:
            raise vRAMError(f"write_reserved: the variable {var_name} cannot be allocated as no memory was reserved for it.")

        self.check_occupied()
        return self.pointer_list[var_name]
            
            

"""To do
wite a dealloc/free function to free memory regions
Check if I can do math operations with byte arrays
implement a simple read function to complement load
update the pointers so that they have an index list with element start:end position lists
Re-tuch the lexer, parse and executor with the additional sections and new opcodes
Implement bitwise operations
"""