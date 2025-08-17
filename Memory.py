
from __future__ import annotations

import struct
from typing import override

from Custom_errors import vRAMError


class Pointer:
    """A Pointer class responsible for representing selected memory blocks as variables and facilitating access to given blocks."""
    def __init__(self, var_name: str, var_type: str, adrs: int, length: int, element_offsets: list[int] | None = None) -> None:
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

    @override
    def __str__(self) -> str:
        return f"\n< Pointer for {self.var_name} of type {self.var_type} >\n< memory address: {self.adrs} >\n< length {self.length} >\n< element offsets {self.element_offsets} >\n< element indexes {self.element_indexes} >\n"
    
    @override
    def __repr__(self) -> str:
        return self.__str__()
    
    def indexer(self, offset_lst: list[int]) -> list[list[int]]:
        """Creates slice-indexes for each element in a memory block/variable"""
        indexes: list[list[int]] = []
        previous: int = self.adrs
        for offset in offset_lst:
            indexes.append([previous, previous + offset])
            previous: int = previous + offset

        self.element_indexes = indexes
        return indexes


class Memory:
    """A Memory class responsible for representing vRAM in the miniAssembly VirtualMachine"""
    def __init__(self, size: int = 1048576, register_supervisor: "RegisterSupervisor | None" = None) -> None:
        self.size: int = size
        self.vram: bytearray = bytearray(size)
        self.pointer_list: dict[str, "Pointer"] = {}
        self.occupied_addresses: list[list[int]] = []

        #Connection to the other components 
        self.register_supervisor: "RegisterSupervisor | None" = register_supervisor


    @override
    def __str__(self) -> str:
        if len(self.occupied_addresses) == 0:
            return f"\n< Virtual memory (vRAM) with {self.size} cells ({self.size/1048576} MB). >\n"
        else:
            return f"\n< Virtual memory (vRAM) with {self.size} cells ({self.size/1048576} MB). >\n< The following addresses are occupied or reserved {self.occupied_addresses} >\n"


    @override
    def __repr__(self) -> str:
        return self.__str__()


    def view_vRAM(self) -> bytearray:
        return self.vram


    def check_occupied(self) -> None:
        """Checks occupied memory regions and updates the connected attribute"""
        output_lst: list[list[int]] = []
        if len(self.pointer_list) != 0:
            for key in self.pointer_list.keys():
                output_lst.append([self.pointer_list[key].adrs,  self.pointer_list[key].adrs + (self.pointer_list[key].length - 1)])
        
        self.occupied_addresses = output_lst


    def view_section(self, ptr: str) -> str:
        """Views a memory block via it's assigned pointer"""
        section_start: int = self.pointer_list[ptr].adrs
        section_end: int = section_start + self.pointer_list[ptr].length
        
        return f"{self.vram[section_start:section_end]}"
    

    def view_pointer(self, ptr_name: str) -> "Pointer":
        return self.pointer_list[ptr_name]
    

    def numeric_to_byte(self, var: int | float, var_type: str) -> bytearray | bytes:
        """Converts standalone numeric data (ints and floats) to bytes"""
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

        raise vRAMError("numeric_to_byte: Unexpected code path reached.")


    def array_to_byte(self, obj: list[int | float], array_type: str) -> tuple[bytearray, list[int]]:
        """Converts arrayed numeric data into a bytearray"""
        byte_object: bytearray | bytes = bytearray()
        byte_buffer: bytearray | bytes = bytearray()
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

        elif isinstance(obj[0], (float)) and array_type == "farray":
            expected_type: type = float
            var_type: str = "float"
        
        else:
            raise vRAMError("array_to_byte: Unexpected array type received.")

        for element in obj:
            if not isinstance(element, (expected_type)):
                raise vRAMError(f"array_to_bytes: mixed element type in array, an element is of {type(element)}, when {expected_type} was expected") 

        for element in obj:
            if isinstance(element, (int)):
                byte_object: bytearray | bytes = self.numeric_to_byte(var = element, var_type = var_type)
                
            elif isinstance(element, (float)):
                byte_object: bytearray | bytes = self.numeric_to_byte(var = element, var_type = var_type)

            byte_buffer.extend(byte_object)
            element_offsets.append(len(byte_object)) #element_offsets are buffer offsets

        return byte_buffer, element_offsets


    def byte_to_numeric(self, var: bytearray | bytes, var_type: str) -> int | float:
        """Converts bytes and bytearrays into numeric data"""
        if var_type == "int":
            if len(var) == 1: #8 bits, little-endian
                frm: str = "<b"
            
            elif len(var) == 2: #16 bits, little-endian
                frm: str = "<h"

            elif len(var) == 4: #32 bits, little-endian
                frm: str = "<i"

            elif len(var) == 8: #64 bits, little-endian
                frm: str = "<q"

            else:
                raise vRAMError(f"byte_to_numeric: The supplied variable: {var} must have 1-8 bytes, got {len(var)}.")

            return struct.unpack(frm, var)[0]
        
        elif var_type == "float":
            if len(var) == 4: #32 bits, little-endian
                frm: str = "<f"

            elif len(var) == 8: #64 bits, little-endian
                frm: str = "<d"

            else:
                raise vRAMError(f"byte_to_numeric: The supplied variable: {var} must have 1-8 bytes, got {len(var)}.")

            return struct.unpack(frm, var)[0]
        
        else:
            raise vRAMError(f"byte_to_numeric: expecting type {int | float}, got type {var_type}.")


    def byte_to_array(self, var_name: str) -> list[int | float]:
        """Converts a bytearray to an array of numeric data"""
        block_start: int = self.pointer_list[var_name].adrs
        block_end: int = block_start + self.pointer_list[var_name].length
        buffer: bytearray = self.vram[block_start : block_end]
        element_indexes: list[list[int]] = self.pointer_list[var_name].element_indexes
        array_out: list[int | float] = []

        for element in element_indexes:
            if self.pointer_list[var_name].var_type == "iarray":
                var: int | float = self.byte_to_numeric(var = buffer[element[0] : element[1]], var_type = "int")
                array_out.append(var)

            elif self.pointer_list[var_name].var_type == "farray":
                var: int | float = self.byte_to_numeric(var = buffer[element[0] : element[1]], var_type = "float")
                array_out.append(var)

        return array_out


    def bytes_to_char_string(self, var: bytearray | bytes, var_type: str) -> str:
        """Converts bytes or a bytearray to character or string data"""
        value_out: str = ""

        if isinstance(var, (bytearray, bytes)) and var_type in {"char", "string"}:
            value_out: str = var.decode(encoding = "iso-8859-2")

        return value_out


    def store(self, var_name: str, obj: int | float | str | list[int | float] | "Register", obj_type: str, adrs: int) -> "Pointer":
        """Static storage function in order to store register values in the vRAM one by one in the available memory (variables) - can be called by the RegisterSupervisor class"""
        if adrs < 0 or adrs > self.size:
            raise vRAMError(f"store: The supplied memory address is out of bounds: address {adrs}, vRAM addresses: {0} to {len(self.vram)}")
        
        else:
            if isinstance(obj, (int, float)) and (obj_type == "int" or obj_type == "float"):
                buffer: bytearray | bytes = self.numeric_to_byte(var = obj, var_type = obj_type)
                obj_len: int = len(buffer)
                
                self.vram[adrs:adrs + obj_len] = buffer
                self.pointer_list[var_name] = Pointer(var_name = var_name, 
                                                var_type = obj_type,
                                                adrs = adrs,
                                                length = obj_len)
                
            elif isinstance(obj, (list)) and (obj_type == "iarray" or obj_type == "farray"): 
                buffer: bytearray | bytes = bytearray()
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
                    self.pointer_list[var_name].indexer(offset_lst = self.pointer_list[var_name].element_offsets)

                    
            elif isinstance(obj, (str)) and (obj_type == "char" or obj_type == "string"):
                buffer: bytearray | bytes = obj.encode(encoding = "iso-8859-2")
                obj_len: int = len(buffer)

                if adrs + obj_len > len(self.vram):
                    raise vRAMError(f"stor: The supplied char or string is out of bounds: array len: {obj_len}, vRAM len: {len(self.vram)}")
                
                else:
                    self.vram[adrs : adrs + obj_len] = buffer
                    self.pointer_list[var_name] = Pointer(var_name = var_name, 
                                                    var_type = obj_type,
                                                    adrs = adrs,
                                                    length = obj_len)
                
            
            ptr: "Pointer" = self.pointer_list[var_name]
            
            self.check_occupied()
            return ptr


    def reserve_over_limit(self, size: int, block_name: str) -> "Pointer":
        """Allocates and reserves additional memory for the vRAM and creates a null pointer"""
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
        
        ptr: "Pointer" = self.pointer_list[block_name]

        return ptr


    def reserve(self, size: int, adrs: int, block_name: str, overwrite: bool = False) -> "Pointer":
        """Reserves a block of existing memory as a variable and creates a null pointer"""
        if size <= 0:
            raise vRAMError(f"reserve: the size of the reserved block must be positive and larger than 0 bytes, {size} were provided.")

        if block_name in self.pointer_list.keys():
            raise vRAMError(f"reserve: the block {block_name} is already reserved or occupied. Free the occupied block first before a new reserve.")

        if overwrite == False:
            for range in self.occupied_addresses:
                if adrs in range:
                    raise vRAMError(f"reserve: the given address {adrs} is already reserved or occupied. Free the occupied block first before a new reserve.")

            for block in self.occupied_addresses:
                block_start: int = block[0]
                block_end: int = block[1]
                
                if block_start < (adrs + size) and block_end >= adrs:
                    raise vRAMError(f"reserve: the given memory block overlaps with an already occupied block {block}. Free the occupied block first or shift the new block.")

        self.pointer_list.update({block_name : Pointer(var_name = block_name,
                                                       var_type = "reserved",
                                                       adrs = adrs,
                                                       length = size)})

        self.check_occupied()
        ptr: "Pointer" = self.pointer_list[block_name]

        return ptr


    def dynamic_store(self, var_name: str, src_register: str, obj_type: str) -> "Pointer":
        """Dynamic storage function in order to store register values from a loop/function into a reserved memory region (array) - can be called by the RegisterSupervisor class"""
        if var_name not in self.pointer_list.keys() and self.pointer_list[var_name].var_type != "reserved":
            raise vRAMError(f"dynamic_store: the variable {var_name} cannot be allocated as no memory was reserved for it.")

        block_start: int = self.pointer_list[var_name].adrs
        block_end: int = self.pointer_list[var_name].adrs + self.pointer_list[var_name].length
        
        transfer_buffer: bytearray | bytes = bytearray()
        offset: int = 0
        current_write_window: int = self.pointer_list[var_name].write_window_start

        
        transfer_buffer: bytearray | bytes = self.register_supervisor.read_register_bytes(target_register = src_register)[0]
        offset = len(transfer_buffer)

        if current_write_window + offset > block_end:
            raise vRAMError(f"dynamic store: the next current element form register {src_register} would exceed the length capacity of the reserved memory block.")
            
        self.vram[current_write_window : current_write_window + offset] = transfer_buffer
        self.pointer_list[var_name].write_window_start = current_write_window + offset
        self.pointer_list[var_name].element_offsets.append(offset)
        if obj_type == "int":
            self.pointer_list[var_name].var_type = "iarray"
        
        elif obj_type == "float":
            self.pointer_list[var_name].var_type = "farray"

        elif obj_type in {"char", "string"}:
            self.pointer_list[var_name].var_type = "string"

        self.pointer_list[var_name].indexer(offset_lst = self.pointer_list[var_name].element_offsets)

        ptr: "Pointer" = self.pointer_list[var_name]
        self.check_occupied()

        return ptr


    def write_reserved(self, var_name: str, obj: int | float | str | list[int | float], obj_type: str) -> "Pointer":
        """Writes data into an existing and reserved memory block/variable and initializes the connected null pointer"""
        if var_name in self.pointer_list.keys() and self.pointer_list[var_name].var_type == "reserved":
            if isinstance(obj, (int, float)) and obj_type in ("int", "float"):
                buffer: bytearray | bytes = self.numeric_to_byte(var = obj, var_type = obj_type)
                block_start: int = self.pointer_list[var_name].adrs
                block_end: int = block_start + len(buffer)

                if len(buffer) > self.pointer_list[var_name].length:
                    raise vRAMError("data length exceeds reserved block size")
                
                else:
                    self.vram[block_start : block_end] = buffer
                    self.pointer_list[var_name].var_type = obj_type

            elif isinstance(obj, (list)) and len(obj) > 0 and obj_type in ("iarray", "farray"):
                buffer: bytearray | bytes = bytearray()
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
                    self.pointer_list[var_name].element_indexes = self.pointer_list[var_name].indexer(offset_lst = new_offsets)


            elif isinstance(obj, (str)) and len(obj) > 0 and obj_type in ("char", "string"):
                buffer: bytearray | bytes = obj.encode(encoding = "iso-8859-2")
                block_start: int = self.pointer_list[var_name].adrs
                block_end: int = block_start + len(buffer)

                if len(buffer) > self.pointer_list[var_name].length:
                    raise vRAMError("data length exceeds reserved block size")
                
                else:
                    self.vram[block_start : block_end] = buffer
                    self.pointer_list[var_name].var_type = obj_type

            else:
                raise vRAMError(f"write_reserved: unknown variable type {obj_type} or empty variable {obj}.")

        else:
            raise vRAMError(f"write_reserved: the variable {var_name} cannot be allocated as no memory was reserved for it.")

        self.check_occupied()
        return self.pointer_list[var_name]


    def free_region(self, var_name: str) -> "Pointer":
        """Returns a memory block/variable belonging to the selected pointer by setting it to 0 and removes the associated pointer"""
        block_start: int = self.pointer_list[var_name].adrs
        block_end: int = block_start + self.pointer_list[var_name].length
        length: int = self.pointer_list[var_name].length

        self.vram[block_start : block_end] = bytearray(length)
        ptr: "Pointer" = self.pointer_list.pop(var_name)
        self.check_occupied()
        
        return ptr


    def free_all(self) -> None:
        """Returns all allocated and occupied memory to the vRAM by setting them to 0 and removing all pointers"""
        self.vram: bytearray = bytearray(self.size)

        self.pointer_list.clear()
        self.check_occupied()

        print(f"free_all: The total vRAM has been freed, each cell has been reset to 0 and all pointers were dropped.")


    def load_variable(self, var_name: str) -> tuple[bytearray , str]:
        """Retrieves a memory block/variable from the memory - can be called by the RegisterSupervisor"""
        block_start: int = self.pointer_list[var_name].adrs
        block_end: int = block_start + self.pointer_list[var_name].length

        data_type: str = self.pointer_list[var_name].var_type
        return_buffer: bytearray = self.vram[block_start : block_end]
    
        return return_buffer, data_type


    def load_index(self, var_name: str, var_index: int) -> tuple[bytearray, str]:
        """Retrieves elements of a selected memory block/variable via its pointer - can be called by the RegisterSupervisor"""
        block_start: int = self.pointer_list[var_name].adrs
        block_end: int = block_start + self.pointer_list[var_name].length
        
        element_index: list[int] = self.pointer_list[var_name].element_indexes[var_index]
        return_buffer: bytearray = self.vram[element_index[0] : element_index[1]]
        
        if self.pointer_list[var_name].var_type == "iarray":
            element_type: str = "int"

        elif self.pointer_list[var_name].var_type == "farray":
            element_type: str = "float"

        else:
            element_type: str = "char"

        return return_buffer, element_type


    def load_address(self, adrs:int | list[int], return_type: str = "bytearray") -> int | bytearray:
        """Retrieves a single memory cell/variable from the memory - can be called by the RegisterSupervisor"""
        if isinstance(adrs, (int)):
            return_buffer: bytearray = self.vram[slice(adrs, adrs + 1)]
        
        elif isinstance(adrs, (list)):    
            if not all(isinstance(_, (int)) for _ in adrs):
                raise MemoryError(f"load_address: the address argument if lis, must contain integers, however a non-integer element was supplied.")
            
            if len(adrs) != 2:
                raise MemoryError(f"load_address: the address argument if lis, must have length two, a list with {len(adrs)} was supplied.")
            
            return_buffer: bytearray = self.vram[slice(adrs[0], adrs[1])] 

        else:
            raise MemoryError(f"load_address: the adrs argument must be either of type int or list, but {type(adrs)} was provided.")

        if return_type == "bytearray":
            return return_buffer
        
        elif return_type == "int":
            return self.byte_to_numeric(var = return_buffer, var_type = 'int')

        else:
            raise MemoryError(f"load_address: return_type must be either bytearray or int, {type(return_type)} was provided.")
        


from Registers import Register
from Registers import RegisterSupervisor
            


"""To do
wite a dealloc/free function to free memory regions
Check if I can do math operations with byte arrays
implement a simple read function to complement load
update the pointers so that they have an index list with element start:end position lists
Re-tuch the lexer, parse and executor with the additional sections and new opcodes
Implement bitwise operations
"""