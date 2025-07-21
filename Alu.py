from __future__ import annotations

import struct
from typing import override

from Custom_errors import AluError

class AU:
    @override
    def __init__(self) -> None:
        self.external_methods: list[str] = ["add_ints", "sub_ints"]

    @override
    def __str__(self) -> str:
        return f"< The Arithmetic unit responsible for handling integer-wise arithmetic operations and bitwise operations >"
    
    @override
    def __repr__(self) -> str:
        return self.__str__()

    def full_adder(self, input_a: int = 0, input_b: int = 0, carry_in: int = 0) -> tuple[int, int]:
        """
        Hardware-level 1-bit full adder implementation.
        
        Performs binary addition of two single bits plus a carry input,
        producing a sum bit and carry output. This mimics actual CPU
        arithmetic logic unit (ALU) behavior.
        
        Args:
            input_a: First input bit (0 or 1)
            input_b: Second input bit (0 or 1) 
            carry_in: Carry input from previous bit position (0 or 1)
            
        Returns:
            tuple[int, int]: (sum_bit, carry_out)
        """
        
        carry: int = carry_in
        carry_out: int = 0

        bit_sum: int = (input_a ^ input_b) ^ carry
        
        if input_a & input_b:
            carry_out = 1

        elif (input_a | input_b) & carry:
            carry_out = 1

        else:
            carry_out = 0

        return bit_sum, carry_out


    def is_input_negative(self, input_int: int, bit_len: int = 64) -> bool:
        """
        Separates the sign bit and checks if the number is negative or positive.
    
        Args:
            input_int: The integer to check
            bit_len: The bit width to use (default: 64)
        
        Returns:
            A boolean value, True if the number is negative, False if positive.
        """

        sign_mask: int = (1 << (bit_len - 1))
        output: bool = False

        if (input_int & sign_mask) != 0:
            output = True

        return  output
            

    def int_to_bits(self, input_int: int, bit_len: int = 64) -> list[int]:
        """
        Convert an integer to its binary representation as a list of bits.
    
        Args:
            input_int: The integer to convert
            bit_len: The bit width to use (default: 64)
        
        Returns:
            A list of bits, with least significant bit first
        """
        
        width_mask: int = (1 << bit_len) -1
        bit_seq: list[int] = []

        masked_input = input_int & width_mask

        for bit_index in range(bit_len):
            bit_seq.append((masked_input >> bit_index) & 1)

        return bit_seq
    
    def bit_to_int(self, input_bits: list[int]) -> int:
        """
        Convert a list of bits to an integer.
    
        Args:
            input_bits: The bit list to convert
        
        Returns:
            The integer value represented by the bit list.
        """
        
        bit_string: int = 0

        for i, bit in enumerate(input_bits):
            mask: int = bit << i
            bit_string = bit_string | mask

        return bit_string

    
    def twos_complement(self, num: int) -> list[int]:
        """
        Converts a number to it's two's complement representation.
    
        Args:
            num: An integer to convert
        
        Returns:
            A two's complement bit list representation of the given number.
        """

        bit_seq: list[int] = self.int_to_bits(input_int = num)
        complement_seq: list[int] = []
        twoC_seq: list[int] = []
        carry_over: int = 0

        for bit in bit_seq:
            if bit == 0:
                complement_seq.append(1)
            else:
                complement_seq.append(0)

        for i, bit in enumerate(complement_seq):
            new_bit: int = 0
            
            if i == 0:
                new_bit, carry_over = self.full_adder(input_a = bit, carry_in = 1)
                twoC_seq.append(new_bit)

            else:
                
                new_bit, carry_over = self.full_adder(input_a = bit, carry_in = carry_over)
                twoC_seq.append(new_bit)

        return twoC_seq

    def add_ints(self, int_1: int, int_2:int, bit_len: int = 64) -> int:
        i_1: list[int] = self.int_to_bits(input_int = int_1)
        i_2: list[int] = self.int_to_bits(input_int = int_2)
        
        carry_over: int = 0
        new_seq: list[int] = []
        output: int = 0
        msb_in: int = 0


        for bit_index in range(bit_len):
            new_bit: int = 0
            
            if bit_index == (bit_len - 1):
                msb_in = carry_over

            new_bit, carry_over = self.full_adder(input_a = i_1[bit_index], input_b = i_2[bit_index], carry_in = carry_over)

            new_seq.append(new_bit)

        if msb_in != carry_over:
            raise AluError(message="add_ints: overflow detected at the end of int addition")

        output = self.bit_to_int(new_seq)

        #Interpret 'result' as a two's‑complement value in 'width' bits, and return it as a Python signed int (the part after the end checks if 
        #the value is interpreted incorrectly).
        if self.is_input_negative(input_int = output) and (output >= (1 << (bit_len - 1))):
            output = output - (1 << bit_len) #this will shift back a two's complement integer into the proper python representation

        return output
    

    def sub_ints(self, int_1: int, int_2:int, bit_len: int = 64) -> int:
        i_1: list[int] = self.int_to_bits(input_int = int_1)
        i_2: list[int] = self.twos_complement(num = int_2)

        carry_over: int = 0
        new_seq: list[int] = []
        output: int = 0
        msb_in: int = 0


        for bit_index in range(bit_len):
            new_bit: int = 0

            if bit_index == (bit_len - 1):
                msb_in = carry_over
            
            new_bit, carry_over = self.full_adder(input_a = i_1[bit_index], input_b = i_2[bit_index], carry_in = carry_over)

            new_seq.append(new_bit)

        if msb_in != carry_over:
            raise AluError(message="sub_ints: overflow detected at the end of int subtraction")
        
        output = self.bit_to_int(new_seq)

        #Interpret 'result' as a two's‑complement value in 'width' bits, and return it as a Python signed int (the part after the end checks if 
        #the value is interpreted incorrectly).
        if self.is_input_negative(input_int = output) and (output >= (1 << (bit_len - 1))):
            output = output - (1 << bit_len) #this will shift back a two's complement integer into the proper python representation

        return output

class Alu:
    def __init__(self, register_supervisor: "RegisterSupervisor | None" = None) -> None:  
        self.input_table: dict[str, bytearray] = {}


        self.last_op: str = ""
        self.last_output: bytearray = bytearray()
        self.last_output_type: str = ""
        
        self.register_supervisor: "RegisterSupervisor | None" = register_supervisor
        self.AU: AU = AU()

        
        self.numeric_operations: set[str] = {"add", "sub", "mul", "idiv", "mod", "inc", "dec"}
        self.char_operations_out: set[str] = {"add", "sub", "mul", "idiv"}
        self.logical_operations_out: set[str] = {"eq", "neq", "gt", "ge", "lt", "le"}
        self.bitwise_operations_out: set[str] = {"and", "or", "xor", "not", "ls", "rs"}

    @override
    def __str__(self) -> str:
        return f"< ALU: last operation {self.last_op} with an output {self.last_output} of type {self.last_output_type} >"
    
    @override
    def __repr__(self) -> str:
        return self.__str__()
    
    def byte_to_numeric(self, var: bytearray, var_type: str) -> tuple[int | float]:
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
        
        elif var_type == "char":
            if len(var) > 8:
                raise AluError(f"byte_to_numeric: The supplied variable: {var} must have 1-8 bytes, got {len(var)}.")
            
            return (ord(var),)

        else:
            raise AluError(f"byte_to_numeric: expecting type {int | float}, got type {var_type}.")
        

    def numeric_to_byte(self, var: int | float, var_type: str) -> bytes:
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


    def add_sub(self, operation: str, dest: str, operand_1: str | int, operand_2: str | int, op1_type: str, op2_type:str) -> None:
        """Invokes the add_int or sub_int function from the AU subunit and the read_register_bytes method from the RegisterSupervisor if one of the operands in a register,
        to add or subtract the provided operands. The function accepts int and char types ( for char type the value will be the integer representation of a character) to allow
        numeric operations and character transformations.
        
        The function returns none, as it directly writes the sum/difference of the operands, into the destination register where the return type is determined
        by te type of the first operand."""
        
        op1_buffer: bytes | bytearray = bytes()
        op2_buffer: bytes | bytearray = bytes()
        op1_buffer_type: str = ""
        op2_buffer_type: str = ""

        op1_int_value: int = 0
        op2_int_value: int = 0

        operation_result: int = 0
        dest_buffer: bytes | bytearray = bytes()
        dest_type: str = ""

        if not isinstance(dest, (str)) or dest not in self.register_supervisor.ret_register_names():
            raise AluError(f"add_sub: the destination argument must be the name of a valid register, {dest} was given.")
        
        if not isinstance(operation, (str)) or operation not in {"add", "sub"}:
            raise AluError(f"add_sub: the operation argument must be the name of a valid instruction, {dest} was given.")

        if isinstance(operand_1, (str)) and op1_type == "register":
            op1_buffer, op1_buffer_type = self.register_supervisor.read_register_bytes(operand_1)
            op1_int_value = self.byte_to_numeric(var = op1_buffer, var_type = "int")[0]

            if op1_buffer_type not in {"int", "char"}:
                raise AluError(f"add_sub: int addition can only be performed on int and integer representation of char classes, {op1_buffer_type} was provided.")
        
        elif isinstance(operand_1, (int)) and op1_type == "int":
            op1_int_value = operand_1
            op1_buffer_type = "int"

        else:
            raise AluError(f"add_sub: expected type 'register' or 'int' as an operand, {op1_type} was provided.")

        if isinstance(operand_2, (str)) and op2_type == "register":
            op2_buffer, op2_buffer_type = self.register_supervisor.read_register_bytes(operand_2)
            op2_int_value = self.byte_to_numeric(var = op2_buffer, var_type = "int")[0]

            if op2_buffer_type not in {"int", "char"}:
                raise AluError(f"add_sub: int addition can only be performed on int and integer representation of char classes, {op2_buffer_type} was provided.")
        
        elif isinstance(operand_2, (int)) and op2_type == "int":
            op2_int_value = operand_2
            op2_buffer_type = "int"

        else:
            raise AluError(f"add_sub: expected type 'register' or 'int' as an operand, {op2_type} was provided.")
        
        
        if operation == "add":
            operation_result = self.AU.add_ints(int_1 = op1_int_value, int_2 = op2_int_value, bit_len = 64)
        
        elif operation == "sub":
            operation_result = self.AU.sub_ints(int_1 = op1_int_value, int_2 = op2_int_value, bit_len = 64)

        if op1_buffer_type == "char" and operation_result < 0 or operation_result > 255:
            raise AluError(f"add_sub: a character transformation caused an unexpected value {operation_result}. A char value cannot be less than 0 or exceed 255.")
        
        dest_buffer = self.numeric_to_byte(var = operation_result, var_type = "int")
        dest_type = op1_buffer_type

        self.register_supervisor.write_register(target_register = dest, value = dest_buffer, value_type = dest_type)




        
from Registers import RegisterSupervisor