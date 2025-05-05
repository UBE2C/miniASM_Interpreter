from __future__ import annotations

import struct

from Custom_errors import AluError

class Alu:
    def __init__(self) -> None:  
        self.input_table: dict[str, bytearray] = {}


        self.last_op: str = ""
        self.last_output: bytearray = bytearray()
        self.last_output_type: str = ""

        
        self.numeric_operations: set[str] = {"add", "sub", "mul", "idiv", "fdiv", "mod", "inc", "dec"}
        self.string_operations_out: set[str] = {"add", "sub", "mul", "idiv"}
        self.logical_operations_out: set[str] = {"eq", "neq", "gt", "ge", "lt", "le"}
        self.bitwise_operations_out: set[str] = {"and", "or", "xor", "not", "ls", "rs"}

    def __str__(self) -> str:
        return f"< ALU: last operation {self.last_op} with an output {self.last_output} of type {self.last_output_type} >"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def byte_to_numeric(self, var: bytearray, var_type: str) -> int | float:
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
                raise AluError(f"byte_to_numeric: The supplied variable: {var} must have 1-8 bytes, got {len(var)}.")

            return struct.unpack(frm, var)
        
        elif var_type == "float":
            if len(var) == 4: #32 bits, little-endian
                frm: str = "<f"

            elif len(var) == 8: #64 bits, little-endian
                frm: str = "<d"

            else:
                raise AluError(f"byte_to_numeric: The supplied variable: {var} must have 1-8 bytes, got {len(var)}.")

            return struct.unpack(frm, var)
        
        else:
            raise AluError(f"byte_to_numeric: expecting type {int | float}, got type {var_type}.")

    def numeric_to_byte(self, var: int | float, var_type: str) -> bytearray:
        if var_type == "int" and not isinstance(var, (int)):
            raise AluError(f"numeric_to_byte: expected an int based on {var_type}, but {type(var)} was supplied.")
        
        if var_type == "float" and not isinstance(var, (float)):
            raise AluError(f"numeric_to_byte: expected a float based on {var_type}, but {type(var)} was supplied.")

        if var_type not in {"int", "float"}:
            raise AluError(f"numeric_to_byte: unknown variable type was supplied.")
        
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
                raise AluError(f"numeric_to_byte: The supplied variable: {var} exceeds the 64-bit limit.")

            return struct.pack(frm, var)
        
        elif var_type == "float" and isinstance(var, (float)):
            if var >= (-2 ** 32) / 2 and var <= ((2 ** 32) / 2)-1: #32 bits, little-endian
                frm: str = "<f"

            elif var >= (-2 ** 64) / 2 and var <= ((2 ** 64) / 2)-1: #64 bits, little-endian
                frm: str = "<d"

            else:
                raise AluError(f"numeric_to_byte: The supplied value: {var} exceeds the 64-bit limit.")

            return struct.pack(frm, var)


        
    