import struct
from Custom_errors import RegisterError

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
    



    def byte_to_char(self, var: bytearray, var_type: str) -> str:
        if var_type == "char":
            return var.decode(encoding = "iso-8859-2")
        
        else:
            raise AluError(f"byte_to_char: expecting type char, got type {var_type}.")
        
        def char_to_byte(self, var: str, var_type: str) -> str:
        if var_type == "char" and isinstance(var, (str)):
            return var.encode(encoding = "iso-8859-2")
        
        else:
            raise AluError(f"char_to_byte: expecting type char, got type {var_type}.")
        








test_program: str = """
; My first program
mov  a, 5
inc  a
call function
msg  '(5+1)/2 = ', a    ; output message
end

function:
    div  a, 2
    ret
"""

test_program_2: str = """
; My first program
mov  rx0, 5
mov rx1, rx0
mov rx2, a
mov rx3, abc
inc  rx0
call function
msg  '(5+1)/2 = ', rx0 , 'and reg2 char = ', rx2, 'and reg3 str = ', rx3    ; output message
end

function:
    div  rx, 2
    ret
"""


test_1: str = '''
msg 'Hello, world!'
end
'''

test_2: str = '''
mov   a, 5
mov   b, a
mov   c, a
call  proc_fact
call  print
end

proc_fact:
    dec   b
    mul   c, b
    cmp   b, 1
    jne   proc_fact
    ret

print:
    msg   a, '! = ', c ; output text
    ret
'''

test_3: str = """
mov   a, 8            ; value
mov   b, 0            ; next
mov   c, 0            ; counter
mov   d, 0            ; first
mov   e, 1            ; second
call  proc_fib
call  print
end

proc_fib:
    cmp   c, 2
    jl    func_0
    mov   b, d
    add   b, e
    mov   d, e
    mov   e, b
    inc   c
    cmp   c, a
    jle   proc_fib
    ret

func_0:
    mov   b, c
    inc   c
    jmp   proc_fib

print:
    msg   'Term ', a, ' of Fibonacci series is: ', b        ; output text
    ret
"""

test_4: str = '''
mov   a, 11           ; value1
mov   b, 3            ; value2
call  mod_func
msg   'mod(', a, ', ', b, ') = ', d        ; output
end

; Mod function
mod_func:
    mov   c, a        ; temp1
    div   c, b
    mul   c, b
    mov   d, a        ; temp2
    sub   d, c
    ret
'''

instr_set: set[str] = {"mov", "inc", "dec", "add", "sub", "mul", "div", "fdiv", ":", "jmp",
                                 "cmp", "jne", "je", "jge", "jg", "jle", "jl", "call", "ret", "str", "ldr",
                                 "msg", ";", "end"}

reg_list: set[str] = {f"rx{i}" for i in range(16)}
reg_list.add("MAR")
reg_list.add("MDR")

instr: list[str] = test_program.split("\n")
instr_2: list[str] = test_program_2.split("\n")


class Token:
    def __init__(self, type: str = None, value: str = None) -> None:
        self.type: str = type
        self.value: str = value
        

    def __str__(self) -> str:
        return f"Token(type = {self.type}, value = {self.value})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
def lexer(instructions: list[str], instruction_set: list[str], register_names: set[str]) -> list[Token]:
    token_lst: list[Token] = []


    def operand_classifier(operand: str) -> Token:
        if operand.isdigit():
            return Token(type = "INT", value = operand)
                                
        elif operand.find(".") != -1:
            return Token(type = "FLOAT", value = operand)
        
        elif operand in register_names:
            return Token(type = "REGISTER", value = operand)
        
        elif len(operand) == 1 and not operand.isdigit():
            return Token(type = "CHAR", value = operand)
        
        elif len(operand) > 1 and operand not in register_names:
            return Token(type = "STRING", value = operand)

        else:
            return Token(type = "IDENT", value = operand)
    

    for i, code_line in enumerate(instructions):
        if not code_line:
                continue
        
        instruction: str = code_line.strip()

        if  instruction.startswith(";"):
            continue

        elif instruction.find(";") != -1:
            instruction = instruction.split(";", 1)[0]
            
        if  instruction.endswith(":"):
            label_end: int =  instruction.find(":")
                
            token_lst.append(Token(type = "LABEL", value =  instruction[0 : label_end]))
        
        elif instruction == "end" or  instruction == "ret":
            token_lst.append(Token(type = "OPCODE", value =  instruction))

        else:
            opcode: str = instruction.split(" ", 1)[0]

            if opcode in instruction_set and opcode not in [";", ":", "end", "ret"]:
                operands: str = instruction.split(" ", 1)[1].strip()

                if opcode == "msg" and "," in operands:
                    token_lst.append(Token(type = "OPCODE", value = "msg"))
                    operand_lst: list[str] = []
                    char_pointer: int = 0
                    
                    while char_pointer < len(operands):
                        
                        if operands[char_pointer] == "'" or operands[char_pointer] == '"':
                            string_end: str = operands[char_pointer]
                            sub_char_p: int = char_pointer + 1
                            message: str = ""
                            
                            while sub_char_p < len(operands) and operands[sub_char_p] != string_end:
                                message += operands[sub_char_p]
                                
                                sub_char_p += 1

                            operand_lst.append("'" + message + "'")
                            char_pointer = sub_char_p
                        
                        else:
                            string_marker: set = {"'", '"'}
                            sub_char_p: int = char_pointer
                            sub_string: str = ""

                            while sub_char_p < len(operands) and operands[sub_char_p] not in string_marker:
                                sub_string += operands[sub_char_p]

                                sub_char_p += 1

                            operand_lst.append(sub_string)
                            char_pointer = sub_char_p -1

                        char_pointer += 1
                    
                    for operand in operand_lst:
                        if operand.startswith("'"):
                            token_lst.append(Token(type = "MESSAGE", value = operand))
                        
                        else:
                            token_lst.append(operand_classifier(operand = operand.replace(",", "").strip()))
                        
                elif opcode == "msg" and "," not in operands:
                    token_lst.append(Token(type = "OPCODE", value = "msg"))
                    
                    if operands.startswith("'") or operands.startswith('"'):
                        message_start: str = operands[0]
                    
                        token_lst.append(Token(type = "MESSAGE", value = "'" + operands.strip(message_start).strip() + "'"))

                    else:
                        token_lst.append(operand_classifier(operand = operands.strip()))

                else:
                    token_lst.append(Token(type = "OPCODE", value = opcode))
                    operands = operands.replace(" ", "")

                    if "," in operands:
                        operand_lst = operands.split(",")
                        
                        for e in range(len(operand_lst)):
                            token_lst.append(operand_classifier(operand = operand_lst[e]))

                    else:
                        token_lst.append(operand_classifier(operand = operands))

            elif opcode not in instruction_set:
                raise ValueError(f"The given opcode {opcode} is not part of the valid instruction set.")

    return token_lst

class Instruction:

    def __init__(self, opcode: str, args: list[str | int | float], arg_types: str = None):
        self.opcode: str = opcode
        self.args: list[str | int | float] = args
        self.arg_types: str = arg_types

    def __str__(self) -> None:
        return f"Instruction(opcode = {self.opcode}, args = {self.args}, arg_types = {self.arg_types})"
    
    def __repr__(self) -> None:
        return self.__str__()
    

def parser(token_list: list[Token]) -> tuple[list[Instruction], dict[str, int]]:
    token_lst: list[Token] = token_list.copy()
    instruction_list: list[Instruction] = []
    jump_table: dict[str, int] = {}
    
    list_pointer: int = 0
    while list_pointer < len(token_lst):
        current_opcode: str = ""
        
        if token_lst[list_pointer].type in ("OPCODE", "LABEL"):
            if token_lst[list_pointer].type == "OPCODE":
                current_opcode = token_lst[list_pointer].value
            
            elif token_lst[list_pointer].type == "LABEL":
                jump_table[token_lst[list_pointer].value] = len(instruction_list)
                list_pointer += 1
                continue
            
            list_pointer += 1
            
            current_args: list[str | int | float] = []
            current_types: list[str] = []
            
            while list_pointer < len(token_lst) and token_lst[list_pointer].type not in ("OPCODE", "LABEL"):
        
                if token_lst[list_pointer].type == "INT":
                    current_args.append(int(token_lst[list_pointer].value))
                    current_types.append("int")
                    list_pointer += 1
                        
                    
                elif token_lst[list_pointer].type == "FLOAT":
                    current_args.append(float(token_lst[list_pointer].value))
                    current_types.append("float")
                    list_pointer += 1

                elif token_lst[list_pointer].type == "REGISTER":
                    current_args.append(token_list[list_pointer].value)
                    current_types.append("register")
                    list_pointer += 1

                elif token_list[list_pointer].type == "CHAR":
                    current_args.append(token_list[list_pointer].value)
                    current_types.append("char")
                    list_pointer += 1

                elif token_list[list_pointer].type == "STRING":
                    current_args.append(token_list[list_pointer].value)
                    current_types.append("string")
                    list_pointer += 1

                elif token_lst[list_pointer].type == "IDENT":
                    current_args.append(token_lst[list_pointer].value)
                    current_types.append("ident")
                    list_pointer += 1
                
                elif token_lst[list_pointer].type == "MESSAGE":
                    current_args.append("'" + token_lst[list_pointer].value + "'")
                    current_types.append("message")
                    list_pointer += 1
                
                else:
                    break
            
            instruction_list.append(Instruction(opcode = current_opcode, args = current_args, arg_types = current_types))

    return instruction_list, jump_table


def code_executor(instruction_lst: list[Instruction], jump_tbl: dict[str, int], register_tbl: Registers) -> str | int:
    output_str: str = ""
    call_stack: list[int] = []
    comp_results: dict[str, bool] = {
        "jne" : False,
        "je" : False,
        "jge" : False,
        "jg" : False,
        "jle" : False, 
        "jl" : False
    }
    ip: int = 0 #instruction pointer

    def reset_comp() -> None:
        for key in comp_results:
            comp_results[key] = False

    while ip < len(instruction_lst) and ip >= 0:
        src_value: int | float = 0
        if instruction_lst[ip].opcode == "mov":
            
            if isinstance(instruction_lst[ip].args[1], (int, float)):
                register_tbl.write(instruction_lst[ip].args[0], instruction_lst[ip].args[1])
            
            else:
                src_value = register_tbl.read(instruction_lst[ip].args[1])
                register_tbl.write(instruction_lst[ip].args[0], src_value)

        elif instruction_lst[ip].opcode == "inc":
            src_value = register_tbl.read(instruction_lst[ip].args[0])
            register_tbl.write(instruction_lst[ip].args[0], src_value + 1)

        elif instruction_lst[ip].opcode == "dec":
            src_value = register_tbl.read(instruction_lst[ip].args[0])
            register_tbl.write(instruction_lst[ip].args[0], src_value - 1)

        elif instruction_lst[ip].opcode == "add":
            
            if isinstance(instruction_lst[ip].args[1], (int, float)):
                src_value = register_tbl.read(instruction_lst[ip].args[0])
                register_tbl.write(instruction_lst[ip].args[0], src_value + instruction_lst[ip].args[1])
            
            else:
                src_value = register_tbl.read(instruction_lst[ip].args[0])
                add_value: int | float = register_tbl.read(instruction_lst[ip].args[1])
                register_tbl.write(instruction_lst[ip].args[0], src_value + add_value)

        elif instruction_lst[ip].opcode == "sub":
            
            if isinstance(instruction_lst[ip].args[1], (int, float)):
                src_value = register_tbl.read(instruction_lst[ip].args[0])
                register_tbl.write(instruction_lst[ip].args[0], src_value - instruction_lst[ip].args[1])
            
            else:
                src_value = register_tbl.read(instruction_lst[ip].args[0])
                sub_value: int | float = register_tbl.read(instruction_lst[ip].args[1])
                register_tbl.write(instruction_lst[ip].args[0], src_value - sub_value)

        elif instruction_lst[ip].opcode == "mul":
            
            if isinstance(instruction_lst[ip].args[1], (int, float)):
                src_value = register_tbl.read(instruction_lst[ip].args[0])
                register_tbl.write(instruction_lst[ip].args[0], src_value * instruction_lst[ip].args[1])
            
            else:
                src_value = register_tbl.read(instruction_lst[ip].args[0])
                mul_value: int | float = register_tbl.read(instruction_lst[ip].args[1])
                register_tbl.write(instruction_lst[ip].args[0], src_value * mul_value)

        elif instruction_lst[ip].opcode == "div":
            
            if isinstance(instruction_lst[ip].args[1], (int, float)) and instruction_lst[ip].args[1] != 0:
                src_value = register_tbl.read(instruction_lst[ip].args[0])
                register_tbl.write(instruction_lst[ip].args[0], src_value // instruction_lst[ip].args[1])
            
            elif isinstance(instruction_lst[ip].args[1], (str)) and register_tbl.get(instruction_lst[ip].args[1]) != 0:
                src_value = register_tbl.read(instruction_lst[ip].args[0])
                div_value: int | float = register_tbl.read(instruction_lst[ip].args[1])
                register_tbl.write(instruction_lst[ip].args[0], src_value // div_value)
            
            else:
                print("Division by 0 is not comprehensible.")
                return -1

        elif instruction_lst[ip].opcode == "fdiv":
            
            if isinstance(instruction_lst[ip].args[1], (int, float)) and instruction_lst[ip].args[1] != 0:
                src_value = register_tbl.read(instruction_lst[ip].args[0])
                register_tbl.write(instruction_lst[ip].args[0], src_value / instruction_lst[ip].args[1])
            
            elif isinstance(instruction_lst[ip].args[1], (str)) and register_tbl.get(instruction_lst[ip].args[1]) != 0:
                src_value = register_tbl.read(instruction_lst[ip].args[0])
                fdiv_value: int | float = register_tbl.read(instruction_lst[ip].args[1])
                register_tbl.write(instruction_lst[ip].args[0], src_value / fdiv_value)
            
            else:
                print("Division by 0 is not comprehensible.")
                return -1

        elif instruction_lst[ip].opcode == "cmp":
            reset_comp()
            arg1: int | float = 0
            arg2: int | float = 0
            
            if isinstance(instruction_lst[ip].args[0], (int, float)) and isinstance(instruction_lst[ip].args[1], (int, float)):
                arg1 = instruction_lst[ip].args[0]
                arg2 = instruction_lst[ip].args[1]

            elif isinstance(instruction_lst[ip].args[0], (str)) and isinstance(instruction_lst[ip].args[1], (int, float)):
                arg1 = register_tbl.read(instruction_lst[ip].args[0])
                arg2 = instruction_lst[ip].args[1]

            elif isinstance(instruction_lst[ip].args[0], (int, float)) and isinstance(instruction_lst[ip].args[1], (str)):
                arg1 = instruction_lst[ip].args[0]
                arg2 = register_tbl.read(instruction_lst[ip].args[1])
            
            elif isinstance(instruction_lst[ip].args[0], (str)) and isinstance(instruction_lst[ip].args[1], (str)):
                arg1 = register_tbl.read(instruction_lst[ip].args[0])
                arg2 = register_tbl.read(instruction_lst[ip].args[1])

            if arg1 != arg2:
                comp_results["jne"] = True

            if arg1 > arg2:
                    comp_results["jg"] = True
            
            if arg1 < arg2:
                    comp_results["jl"] = True

            if arg1 == arg2:
                comp_results["je"] = True

            if arg1 >= arg2:
                comp_results["jge"] = True
            
            if arg1 <= arg2:
                comp_results["jle"] = True 
               
            
        elif instruction_lst[ip].opcode == "jmp":
            ip = jump_tbl.get(instruction_lst[ip].args[0])
            continue

        elif instruction_lst[ip].opcode in {"jne", "je", "jge", "jg", "jle", "jl"}:
            if comp_results.get(instruction_lst[ip].opcode) == True:
                ip = jump_tbl.get(instruction_lst[ip].args[0])
                
                continue
            else:
                ip += 1
                continue

        elif instruction_lst[ip].opcode == "call":
            call_stack.append(ip)
            ip = jump_tbl.get(instruction_lst[ip].args[0])
            continue

        elif instruction_lst[ip].opcode == "ret":
            if len(call_stack) > 0:
                ip = call_stack.pop() + 1
                continue

            else:
                return -1
            
        elif instruction_lst[ip].opcode == "msg":
            for arg in instruction_lst[ip].args:
                if isinstance(arg, (str)) and arg.startswith("'"):
                    output_arg: str = arg.strip("'")
                    output_str += output_arg
                    
                elif isinstance(arg, (str)):
                    if isinstance(register_tbl.read(arg), (float)) and register_tbl.read(arg).is_integer():
                        output_str += str(int(register_tbl.read(arg)))
                    else:
                        output_str += str(register_tbl.read(arg))
                    
        elif instruction_lst[ip].opcode == "end":
            return output_str.strip()
        
        else:
            break
        
        ip += 1

    return -1


"""Original register implementation"""
from Custom_errors import RegisterError
import struct

class Registers:
    def __init__(self, number: int = 16):
        self.registers: dict[str, bytearray] = {f"rx{i}": bytearray(8) for i in range(number)}
        self.registers.update({"MAR" : bytearray(8)})  # Memory Address Register
        self.registers.update({"MDR" : bytearray(8)})  # Memory Data Register

    def __str__(self) -> str:
        return f"< register names: {self.registers.keys()} >\n< register values: {self.registers.values()} >"

    def __repr__(self) -> str:
        return self.__str__()

# extend the read an write functions so they can convert int, float and str into the appropriate byte arrays
    def read(self, name: str) -> bytearray:
        return self.registers.get(name)
        
    def write(self, name: str, value: bytearray) -> None:
        if len(value) <= 8:
            self.registers[name] = value
        else:
            raise RegisterError(f"write: register value must be maximum 64 bits, register {name} got {len(value)}.")
        
# implement the load_ind and load_adr function in order to be able to fetch data from the vRAM either based on address or address and var index
    def names(self) -> set[str]:
        return set(self.registers.keys())
    



class Register:
    def __init__(self, name: str, size: int = 8, register_group: dict[str, object] = None):
        self.name: str = name
        self.limit: int = size
        if register_group ==  None:
            self.register_group: dict[str, object] = {}
        else:
            self.register_group = register_group
        self.bytes: bytearray =  bytearray(size)
        self.value: int | float | str | bool = self.read_value()
        self.data_type: str = "Empty"

    def __str__(self) -> str:
        return f"< Register {self.name}, capacity: {self.limit} bytes, value: {self.bytes}, value type: {self.data_type}>"
    
    def __repr__(self) -> str:
        return self.__str__()


    def numeric_to_byte(self, var: int | float, var_type: str) -> bytearray:
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

    def boolean_to_byte(self, var: bool, var_type: str) -> bytearray:
        buffer_out: bytearray = bytearray()

        if isinstance(var, (bool)) and var_type == "bool":
            buffer_out = var.to_bytes(byteorder = "little")

        return buffer_out
    
    def char_string_to_bytes(self, var: str, var_type: str) -> bytearray:
        buffer_out: bytearray = bytearray()

        if isinstance(var, (str)) and var_type in {"char", "string"}:
            buffer_out = var.encode(encoding = "iso-8859-2")

        return buffer_out


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
        
    def bytes_to_boolean(self, var: bytearray, var_type: str) -> bool:
        value_out: bool = False

        if isinstance(var, (bytearray)) and var_type == "bool":
            value_out = struct.unpack("<?", var)[0]

        return value_out
    
    def bytes_to_char_string(self, var: bytearray, var_type: str) -> str:
        value_out: str = ""

        if isinstance(var, (bytearray)) and var_type in {"char", "string"}:
            value_out = var.decode(encoding = "iso-8859-2")

        return value_out


    def read_bytes(self) -> bytearray:
        return self.bytes
    
    def read_value(self) -> int | float | str | bool:
        value_out_numeric: int | float = 0
        value_out_str: str = ""
        value_out_bool: bool = False

        if self.data_type in {"int", "float"}:
            value_out_numeric = self.byte_to_numeric(var = self.bytes, var_type = self.data_type)
            return value_out_numeric

        elif self.data_type in {"char", "string"}:
            value_out_str = self.bytes_to_char_string(var = self.bytes, var_type = self.data_type)
            return value_out_str

        elif self.data_type == "bool":
            value_out_bool = self.bytes_to_boolean(var = self.bytes, var_type = self.data_type)
            return value_out_bool
        
        else:
            value_out_numeric = 0
            return value_out_numeric

    def write(self, value: int | float | str | bool, value_type: str) -> None:
        byte_buffer: bytearray = bytearray(self.limit)

        if isinstance(value, (int, float)) and value_type in {"int", "float"}:
            byte_buffer = self.numeric_to_byte(var = value, var_type = value_type)

            if len(byte_buffer) > 8:
                RegisterError(f"write: register {self.name} got a value {len(value)} exceeding it's limit {self.limit}.")
            
            self.bytes = byte_buffer
            self.data_type = value_type

        elif isinstance(value, (str)) and value_type == "register":
            byte_buffer = register_list.get(value).read_bytes()
            
            if len(byte_buffer) > 8:
                RegisterError(f"write: register {self.name} got a value {len(value)} exceeding it's limit {self.limit}.")

            self.bytes = byte_buffer
            self.data_type = register_list.get(value).data_type

        elif isinstance(value, (str)) and value_type in {"char", "string"}:
            byte_buffer = self.char_string_to_bytes(var = value, var_type = value_type)

            if len(byte_buffer) > 8:
                RegisterError(f"write: register {self.name} got a value {len(value)} exceeding it's limit {self.limit}.")

            self.bytes = byte_buffer
            self.data_type = value_type

        elif isinstance(value, (bool)) and value_type == "bool":
            byte_buffer = self.boolean_to_byte(var = value, var_type = value_type)

            if len(byte_buffer) > 8:
                RegisterError(f"write: register {self.name} got a value {len(value)} exceeding it's limit {self.limit}.")

            self.bytes = byte_buffer
            self.data_type = value_type

        self.value = self.read_value()


"""Think about creating a register manager to manage each register and make tem aware of each other"""

register_list: dict[str, Register] = {f"rx{i}" : Register(name = f"rx{i}", size = 8, register_group = register_list) for i in range(16)}

        

        
""" 'Self aware' register setup:
each register is aware of the group they a re a part of and they can pass values among each other once the group was initialized
and the registers were connected"""

class Register:
    def __init__(self, name: str, size: int = 8, register_group: dict[str, object] = None):
        self.name: str = name
        self.limit: int = size
        if register_group ==  None: #allows the register to recognize the other registers
            self.register_group: dict[str, object] = {}
        else:
            self.register_group = register_group
        self.bytes: bytearray =  bytearray(size)
        self.data_type: str = "Empty"
        self.value: int | float | str | bool = self.read_value()
        

    def __str__(self) -> str:
        return f"< Register {self.name}, capacity: {self.limit} bytes, value: {self.bytes}, value type: {self.data_type}>"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def recognize_group(self, group_name: str) -> None:
        self.register_group = group_name


    def numeric_to_byte(self, var: int | float, var_type: str) -> bytearray:
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

    def boolean_to_byte(self, var: bool, var_type: str) -> bytearray:
        buffer_out: bytearray = bytearray()

        if isinstance(var, (bool)) and var_type == "bool":
            buffer_out = var.to_bytes(byteorder = "little")

        return buffer_out
    
    def char_string_to_bytes(self, var: str, var_type: str) -> bytearray:
        buffer_out: bytearray = bytearray()

        if isinstance(var, (str)) and var_type in {"char", "string"}:
            buffer_out = var.encode(encoding = "iso-8859-2")

        return buffer_out


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
        
    def bytes_to_boolean(self, var: bytearray, var_type: str) -> bool:
        value_out: bool = False

        if isinstance(var, (bytearray)) and var_type == "bool":
            value_out = struct.unpack("<?", var)[0]

        return value_out
    
    def bytes_to_char_string(self, var: bytearray, var_type: str) -> str:
        value_out: str = ""

        if isinstance(var, (bytearray)) and var_type in {"char", "string"}:
            value_out = var.decode(encoding = "iso-8859-2")

        return value_out


    def read_bytes(self) -> bytearray:
        return self.bytes
    
    def read_value(self) -> int | float | str | bool:
        value_out_numeric: int | float = 0
        value_out_str: str = ""
        value_out_bool: bool = False

        if self.data_type in {"int", "float"}:
            value_out_numeric = self.byte_to_numeric(var = self.bytes, var_type = self.data_type)
            return value_out_numeric

        elif self.data_type in {"char", "string"}:
            value_out_str = self.bytes_to_char_string(var = self.bytes, var_type = self.data_type)
            return value_out_str

        elif self.data_type == "bool":
            value_out_bool = self.bytes_to_boolean(var = self.bytes, var_type = self.data_type)
            return value_out_bool
        
        elif self.data_type == "Empty":
            value_out_numeric = 0
            return value_out_numeric

    def write(self, value: int | float | str | bool, value_type: str) -> None:
        byte_buffer: bytearray = bytearray(self.limit)

        if isinstance(value, (int, float)) and value_type in {"int", "float"}:
            byte_buffer = self.numeric_to_byte(var = value, var_type = value_type)

            if len(byte_buffer) > 8:
                RegisterError(f"write: register {self.name} got a value {len(value)} exceeding it's limit {self.limit}.")
            
            self.bytes = byte_buffer
            self.data_type = value_type

        elif isinstance(value, (str)) and value_type == "register":
            if len(self.register_group) == 0:
                RegisterError(f"write: register {self.name} is not part of a register group and is not connected to other registers.")

            byte_buffer = self.register_group.get(value).read_bytes()
            
            if len(byte_buffer) > 8:
                RegisterError(f"write: register {self.name} got a value {len(value)} exceeding it's limit {self.limit}.")

            self.bytes = byte_buffer
            self.data_type = self.register_group.get(value).data_type

        elif isinstance(value, (str)) and value_type in {"char", "string"}:
            byte_buffer = self.char_string_to_bytes(var = value, var_type = value_type)

            if len(byte_buffer) > 8:
                RegisterError(f"write: register {self.name} got a value {len(value)} exceeding it's limit {self.limit}.")

            self.bytes = byte_buffer
            self.data_type = value_type

        elif isinstance(value, (bool)) and value_type == "bool":
            byte_buffer = self.boolean_to_byte(var = value, var_type = value_type)

            if len(byte_buffer) > 8:
                RegisterError(f"write: register {self.name} got a value {len(value)} exceeding it's limit {self.limit}.")

            self.bytes = byte_buffer
            self.data_type = value_type

        self.value = self.read_value()


"""Think about creating a register manager to manage each register and make tem aware of each other"""
register_list: dict[str, Register] = {}
register_list = {f"rx{i}" : Register(name = f"rx{i}", size = 8, register_group = register_list) for i in range(16)}
for key in register_list:
    register_list[key].recognize_group(register_list)







""" Managed register setup:
the registers are not aware of each other, and the read, write and transfer functions are managed on the group level by the 
group control"""

class Register:
    def __init__(self, name: str, size: int = 8):
        self.name: str = name
        self.limit: int = size
        self.bytes: bytearray =  bytearray(size)
        self.data_type: str = "Empty"
        self.value: int | float | str | bool = self.read_value()
        

    def __str__(self) -> str:
        return f"< Register {self.name}, capacity: {self.limit} bytes, value: {self.bytes}, value type: {self.data_type}>"
    
    def __repr__(self) -> str:
        return self.__str__()


    def numeric_to_byte(self, var: int | float, var_type: str) -> bytearray:
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

    def boolean_to_byte(self, var: bool, var_type: str) -> bytearray:
        buffer_out: bytearray = bytearray()

        if isinstance(var, (bool)) and var_type == "bool":
            buffer_out = var.to_bytes(byteorder = "little")

        return buffer_out
    
    def char_string_to_bytes(self, var: str, var_type: str) -> bytearray:
        buffer_out: bytearray = bytearray()

        if isinstance(var, (str)) and var_type in {"char", "string"}:
            buffer_out = var.encode(encoding = "iso-8859-2")

        return buffer_out


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
        
    def bytes_to_boolean(self, var: bytearray, var_type: str) -> bool:
        value_out: bool = False

        if isinstance(var, (bytearray)) and var_type == "bool":
            value_out = struct.unpack("<?", var)[0]

        return value_out
    
    def bytes_to_char_string(self, var: bytearray, var_type: str) -> str:
        value_out: str = ""

        if isinstance(var, (bytearray)) and var_type in {"char", "string"}:
            value_out = var.decode(encoding = "iso-8859-2")

        return value_out


    def read_bytes(self) -> bytearray:
        return self.bytes
    
    def read_value(self) -> int | float | str | bool:
        value_out_numeric: int | float = 0
        value_out_str: str = ""
        value_out_bool: bool = False

        if self.data_type in {"int", "float"}:
            value_out_numeric = self.byte_to_numeric(var = self.bytes, var_type = self.data_type)
            return value_out_numeric

        elif self.data_type in {"char", "string"}:
            value_out_str = self.bytes_to_char_string(var = self.bytes, var_type = self.data_type)
            return value_out_str

        elif self.data_type == "bool":
            value_out_bool = self.bytes_to_boolean(var = self.bytes, var_type = self.data_type)
            return value_out_bool
        
        elif self.data_type == "Empty":
            value_out_numeric = 0
            return value_out_numeric

    def write(self, value: int | float | str | bool, value_type: str) -> None:
        byte_buffer: bytearray = bytearray(self.limit)

        if isinstance(value, (int, float)) and value_type in {"int", "float"}:
            byte_buffer = self.numeric_to_byte(var = value, var_type = value_type)

            if len(byte_buffer) > 8:
                RegisterError(f"write: register {self.name} got a value {len(value)} exceeding it's limit {self.limit}.")
            
            self.bytes = byte_buffer
            self.data_type = value_type

        elif isinstance(value, (bytearray, bytes)) and value_type in {"int", "float", "char", "string", "bool"}: #register-to-register transfer
            byte_buffer = value
            
            if len(byte_buffer) > 8:
                RegisterError(f"copy_register_value: register {self.name} got a value {len(byte_buffer)} exceeding it's limit {self.limit}.")
            
            self.bytes = byte_buffer
            self.data_type = value_type

        elif isinstance(value, (str)) and value_type in {"char", "string"}:
            byte_buffer = self.char_string_to_bytes(var = value, var_type = value_type)

            if len(byte_buffer) > 8:
                RegisterError(f"write: register {self.name} got a value {len(value)} exceeding it's limit {self.limit}.")

            self.bytes = byte_buffer
            self.data_type = value_type

        elif isinstance(value, (bool)) and value_type == "bool":
            byte_buffer = self.boolean_to_byte(var = value, var_type = value_type)

            if len(byte_buffer) > 8:
                RegisterError(f"write: register {self.name} got a value {len(value)} exceeding it's limit {self.limit}.")

            self.bytes = byte_buffer
            self.data_type = value_type

        self.value = self.read_value()

    def reset_register(self) -> None:
        self.bytes = bytearray(self.limit)
        self.data_type = "Empty"

        self.value = self.read_value()

class RegisterSupervisor:
    def __init__(self) -> None:
        self.register_group: dict[str, Register] = {}
        self.group_size: int = 0
        self.register_byte_limit: int = 8
        
    def __str__(self) -> str:
        return f"< RegisterSupervisor which supervises a group of {len(self.register_group)} registers. >"

    def __repr__(self) -> str:
        return self.__str__()

    def create_register_group(self, register_name_base: str = "rx", register_size: int = 8, group_size: int = 16) -> str:
        self.register_group = {f"{register_name_base}{i}" : Register(name = f"{register_name_base}{i}", size = register_size) for i in range(group_size)}
        self.register_group.update({"mar" : Register(name = "mar", size = register_size)})
        self.register_group.update({"mdr" : Register(name = "mdr", size = register_size)})

        self.group_size = group_size
        self.register_byte_limit = register_size

        return f"A register group of {group_size + 2} registers, from {register_name_base}0 to {register_name_base}{group_size - 1} were created with additional MAR and MDR registers."
        
    def list_registers(self) -> list[str]:
        return list(self.register_group.keys())
    
    def copy_register_value(self, dest_register: str, src_register: str) -> None:
        
        transfer_buffer: bytearray = bytearray()
        source_type: str = ""

        transfer_buffer = self.register_group.get(src_register).read_bytes()
        source_type = self.register_group.get(src_register).data_type
            
        self.register_group.get(dest_register).write(value = transfer_buffer, value_type = source_type)

    def write_register(self, target_register: str, value: int | float | str | bool | Register, value_type: str) -> None:
        if isinstance(self.register_group.get(value), (Register)) and value_type == "register":
            transfer_buffer: bytearray = bytearray()
            source_type: str = ""

            transfer_buffer = self.register_group.get(value).read_bytes()
            source_type = self.register_group.get(value).data_type
            
            self.register_group.get(target_register).write(value = transfer_buffer, value_type = source_type)
            
        else:
            self.register_group.get(target_register).write(value = value, value_type = value_type)

    def read_register_bytes(self, target_register: str) -> tuple[bytearray, str]:
        bytearray_out: bytearray = self.register_group.get(target_register).bytes
        data_type_out: str = self.register_group.get(target_register).data_type
        
        return bytearray_out, data_type_out
    
    def read_register_value(self, target_register: str) -> tuple[int | float | str | bool, str]:
        value_out: int | float | str | bool = self.register_group.get(target_register).value
        data_type_out: str = self.register_group.get(target_register).data_type

        return value_out, data_type_out
    
    def reset_register(self, target_register: str) -> str:
        self.register_group.get(target_register).reset_register()

        return f"Register {self.register_group.get(target_register).name}, has been reset to {self.read_register_bytes(target_register)}"
    
    def reset_all_registers(self) -> str:
        for key in self.register_group:
            self.register_group.get(key).reset_register()

        return f"All register has been reset to {bytearray(self.register_byte_limit)}, Empty."

