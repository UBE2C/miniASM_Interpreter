from __future__ import annotations

import struct
from typing import override

from Custom_errors import RegisterError


class Register:
    def __init__(self, name: str, size: int = 8) -> None:
        self.name: str = name
        self.limit: int = size
        self.bytes: bytearray | bytes =  bytearray(size)
        self.data_type: str = "Empty"
        self.value: int | float | str | bool = self.read_value()

    @override
    def __str__(self) -> str:
        return f"\n< Register {self.name} >\n< capacity: {self.limit} bytes >\n< value: {self.bytes} >\n< value type: {self.data_type} >\n"
    
    @override
    def __repr__(self) -> str:
        return self.__str__()


    def numeric_to_byte(self, var: int | float, var_type: str) -> bytearray | bytes:
        if var_type == "int" and not isinstance(var, (int)):
            raise RegisterError(f"numeric_to_byte: expected an int based on {var_type}, but {type(var)} was supplied.")
        
        if var_type == "float" and not isinstance(var, (float)):
            raise RegisterError(f"numeric_to_byte: expected a float based on {var_type}, but {type(var)} was supplied.")

        if var_type not in {"int", "float"}:
            raise RegisterError(f"numeric_to_byte: unknown variable type was supplied.")
        
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
                raise RegisterError(f"numeric_to_byte: The supplied variable: {var} exceeds the 64-bit limit.")

            return struct.pack(frm, var)
        
        elif var_type == "float" and isinstance(var, (float)):
            if var >= (-2 ** 32) / 2 and var <= ((2 ** 32) / 2)-1: #32 bits, little-endian
                frm: str = "<f"

            elif var >= (-2 ** 64) / 2 and var <= ((2 ** 64) / 2)-1: #64 bits, little-endian
                frm: str = "<d"

            else:
                raise RegisterError(f"numeric_to_byte: The supplied value: {var} exceeds the 64-bit limit.")

            return struct.pack(frm, var)

        raise RegisterError("numeric_to_byte: Unexpected code path reached.")

    def boolean_to_byte(self, var: bool, var_type: str) -> bytearray | bytes:
        buffer_out: bytearray | bytes = bytes()

        if isinstance(var, (bool)) and var_type == "bool":
            buffer_out: bytes = var.to_bytes(byteorder = "little")

        return buffer_out
    
    def char_string_to_bytes(self, var: str, var_type: str) -> bytearray | bytes:
        buffer_out: bytearray | bytes = bytes()

        if isinstance(var, (str)) and var_type in {"char", "string"}:
            buffer_out: bytes = var.encode(encoding = "iso-8859-2")

        return buffer_out


    def byte_to_numeric(self, var: bytearray | bytes, var_type: str) -> int | float:
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
                raise RegisterError(f"byte_to_numeric: The supplied variable: {var} must have 1-8 bytes, got {len(var)}.")

            return struct.unpack(frm, var)[0]
        
        elif var_type == "float":
            if len(var) == 4: #32 bits, little-endian
                frm: str = "<f"

            elif len(var) == 8: #64 bits, little-endian
                frm: str = "<d"

            else:
                raise RegisterError(f"byte_to_numeric: The supplied variable: {var} must have 1-8 bytes, got {len(var)}.")

            return struct.unpack(frm, var)[0]
        
        else:
            raise RegisterError(f"byte_to_numeric: expecting type {int | float}, got type {var_type}.")
        
    def bytes_to_boolean(self, var: bytearray | bytes, var_type: str) -> bool:
        value_out: bool = False

        if isinstance(var, (bytearray)) and var_type == "bool":
            value_out: bool = struct.unpack("<?", var)[0]

        return value_out
    
    def bytes_to_char_string(self, var: bytearray | bytes, var_type: str) -> str:
        value_out: str = ""

        if isinstance(var, (bytearray, bytes)) and var_type in {"char", "string"}:
            value_out: str = var.decode(encoding = "iso-8859-2")

        return value_out


    def read_bytes(self) -> bytearray | bytes:
        return self.bytes
    

    def read_value(self) -> int | float | str | bool:
        value_out_numeric: int | float = 0
        value_out_str: str = ""
        value_out_bool: bool = False

        if self.data_type in {"int", "float"}:
            value_out_numeric: int | float = self.byte_to_numeric(var = self.bytes, var_type = self.data_type)
            return value_out_numeric

        elif self.data_type in {"char", "string"}:
            value_out_str: str = self.bytes_to_char_string(var = self.bytes, var_type = self.data_type)
            return value_out_str

        elif self.data_type == "bool":
            value_out_bool: bool = self.bytes_to_boolean(var = self.bytes, var_type = self.data_type)
            return value_out_bool
        
        elif self.data_type == "Empty":
            value_out_numeric = 0
            return value_out_numeric
        
        raise RegisterError("read_value: Unexpected code path reached.")

    def write(self, value: int | float | str | bool | bytes | bytearray, value_type: str) -> None:
        byte_buffer: bytearray | bytes = bytearray(self.limit)

        if isinstance(value, (int, float)) and value_type in {"int", "float"}:
            byte_buffer: bytearray | bytes = self.numeric_to_byte(var = value, var_type = value_type)

            if len(byte_buffer) > 8:
               raise RegisterError(f"write: register {self.name} got a value {value} exceeding it's limit {self.limit}.")
            
            self.bytes = byte_buffer
            self.data_type = value_type

        elif isinstance(value, (bytearray, bytes)) and value_type in {"int", "float", "char", "string", "bool"}: #register-to-register transfer
            byte_buffer:bytearray | bytes = value
            
            if len(byte_buffer) > 8:
               raise RegisterError(f"copy_register_value: register {self.name} got a value {len(byte_buffer)} exceeding it's limit {self.limit}.")
            
            self.bytes = byte_buffer
            self.data_type = value_type

        elif isinstance(value, (str)) and value_type in {"char", "string"}:
            byte_buffer:bytearray | bytes = self.char_string_to_bytes(var = value, var_type = value_type)

            if len(byte_buffer) > 8:
              raise RegisterError(f"write: register {self.name} got a value {len(value)} exceeding it's limit {self.limit}.")

            self.bytes = byte_buffer
            self.data_type = value_type

        elif isinstance(value, (bool)) and value_type == "bool":
            byte_buffer:bytearray | bytes = self.boolean_to_byte(var = value, var_type = value_type)

            if len(byte_buffer) > 8:
               raise RegisterError(f"write: register {self.name} got a value {value} exceeding it's limit {self.limit}.")

            self.bytes = byte_buffer
            self.data_type = value_type

        self.value = self.read_value()

    def reset_register(self) -> None:
        self.bytes = bytearray(self.limit)
        self.data_type = "Empty"

        self.value = self.read_value()

class RegisterSupervisor:
    def __init__(self) -> None:
        self.register_group: dict[str, "Register"] = {}
        self.group_size: int = 0
        self.register_byte_limit: int = 8
        
        #Connection to the other components 
        self.vRAM: "Memory | None" = None
        self.ALU: "Alu | None" = None
    
    @override
    def __str__(self) -> str:
        return f"\n< RegisterSupervisor which supervises a group of {len(self.register_group)} registers. >\n"

    @override
    def __repr__(self) -> str:
        return self.__str__()

    def create_register_group(self, register_name_base: str = "rx", register_size: int = 8, group_size: int = 16) -> str:
        self.register_group = {f"{register_name_base}{i}" : Register(name = f"{register_name_base}{i}", size = register_size) for i in range(group_size)}
        self.register_group.update({"mar" : Register(name = "mar", size = register_size)})

        self.group_size = group_size
        self.register_byte_limit = register_size

        return f"A register group of {group_size + 2} registers, from {register_name_base}0 to {register_name_base}{group_size - 1} were created with additional MAR and MDR registers."
        
    def list_registers(self) -> list[str]:
        return list(self.register_group.keys())
    
    def copy_register_bytes(self, dest_register: str, src_register: str) -> None:
        transfer_buffer: bytearray | bytes = bytearray()
        source_type: str = ""
        
        transfer_buffer:bytearray | bytes = self.register_group[src_register].read_bytes()
        source_type: str = self.register_group[src_register].data_type
            
        self.register_group[dest_register].write(value = transfer_buffer, value_type = source_type)

    def write_register(self, target_register: str, value: int | float | str | bool, value_type: str) -> None:
        self.register_group[target_register].write(value = value, value_type = value_type)

    def read_register_bytes(self, target_register: str) -> tuple[bytearray | bytes, str]:
        bytearray_out: bytearray | bytes = self.register_group[target_register].bytes
        data_type_out: str = self.register_group[target_register].data_type
        
        return bytearray_out, data_type_out
    
    def read_register_value(self, target_register: str) -> tuple[int | float | str | bool, str]:
        value_out: int | float | str | bool = self.register_group[target_register].value
        data_type_out: str = self.register_group[target_register].data_type

        return value_out, data_type_out
    
    def reset_register(self, target_register: str) -> str:
        self.register_group[target_register].reset_register()

        return f"Register {self.register_group[target_register].name}, has been reset to {self.read_register_bytes(target_register)}"
    
    def reset_all_registers(self) -> str:
        for key in self.register_group:
            self.register_group[key].reset_register()

        return f"All register has been reset to {bytearray(self.register_byte_limit)}, Empty."

    def dstore(self, src_register: str, var_name: str) -> "Pointer":
        self.vRAM.dynamic_store(var_name = var_name, src_register = src_register, obj_type = self.register_group[src_register].data_type)

    def store(self, src_register: str, var_name: str) -> "Pointer":
        self.vRAM.store(var_name = var_name, obj = self.read_register_value(src_register)[0] , obj_type = self.read_register_value(src_register)[1], adrs = self.read_register_value("mar")[0])

    def load_variable(self, )


from Memory import Pointer
from Memory import Memory
from Alu import Alu
