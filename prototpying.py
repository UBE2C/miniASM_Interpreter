import struct
from Custom_errors import AluError, RegisterError

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
    add  rx0, 3
    sub  rx0, 3
    div  rx0, 2
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

"""Updated lexer to process additional types"""

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


"""Updated Instruction class and parser to process the type system"""

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


"""Did not update yet, will need a serious re-work"""

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





from Alu import Alu

#This will be a function inside the VM to initialize the connection between the components
def initialize_connections(self) -> str:
    self.Registers.vRAM: Memory = vRAM
    self.Registers.Alu: Alu = ALU
    self.vRAM.register_supervisor: RegisterSupervisor = Registers
    self.ALU.register_supervisor: RegisterSupervisor = Registers

    return f"Each component has been successfully connected to one another."



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

        #Connection to the other components 
        self.vRAM: Memory = None
        self.ALU: Alu = None
        
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

    def dstore(src_register: str, var_name: str) -> Pointer:
        self.vRAM.dynamic_store(var_name = var_name, src_register = self.register_group.get(src), obj_type = self.register_group.get(src).data_type)

    def store(src_register: str, var_name: str, adrs: int) -> Pointer:
        self.vRAM.store(var_name = var_name, obj = self.register_group.get(src_register), obj_type = self.register_group.get(src).data_type, adrs = self.register_group.get("mar").read_value())    




""" Integration test for the Memory and the registers"""

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

        #Connection to the other components 
        self.register_supervisor: RegisterSupervisor = None

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
            
    """Static storage function in order to store register values in the vRAM one by one in the available memory (variables) - can be called by the RegisterSupervisor class"""
    def store(self, var_name: str, obj: int | float | str | list[int | float | str] | Register, obj_type: str, adrs: int) -> Pointer:
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

    """Dynamic storage function in order to store register values from a loop/function into a reserved memory region (array) - can be called by the RegisterSupervisor class"""
    def dynamic_store(self, var_name: str, src_register: Register, obj_type: str) -> Pointer:
        if var_name not in self.pointer_list.keys() and self.pointer_list[var_name].var_type != "reserved":
            raise vRAMError(f"dynamic_store: the variable {var_name} cannot be allocated as no memory was reserved for it.")

        block_start: int = self.pointer_list.get(var_name).adrs
        block_end: int = self.pointer_list.get(var_name).adrs + self.pointer_list.get(var_name).length
        
        transfer_buffer: bytearray = bytearray()
        offset: int = 0
        current_write_window: int = self.pointer_list.get(var_name).write_window_start

        
        transfer_buffer = self.register_supervisor.register_group.get(src_register).read_bytes()
        offset = len(transfer_buffer)

        if current_write_window + offset > block_end:
            raise vRAMError(f"dynamic store: the next current element form register {src_register.name} would exceed the length capacity of the reserved memory block.")
            
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





def add_overflow_detection(self, int_1: int, int_2:int, result: int) -> bool:
        sign_mask: int = (1<<63)
        output: bool = False

        i_1_sign: bool = (int_1 & sign_mask) != 0
        i_2_sign: bool = (int_2 & sign_mask) != 0
        res_sign: bool = (result & sign_mask) != 0

        if (i_1_sign == i_2_sign) and (res_sign != i_1_sign):
            output = True

        return  output
    
def sub_overflow_detection(self, int_1: int, int_2:int, result: int) -> bool:
    sign_mask: int = (1<<63)
    output: bool = False

    i_1_sign: bool = (int_1 & sign_mask) != 0
    i_2_sign: bool = (int_2 & sign_mask) != 0
    res_sign: bool = (result & sign_mask) != 0

    if (i_1_sign != i_2_sign) and (res_sign == i_2_sign):
        output = True

    return  output





from Custom_errors import AluError
class AU:
    def __init__(self) -> None:
        pass

    def full_adder(self, input_a: int = 0, input_b: int = 0, carry_in: int = 0) -> tuple[int, int]:
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


    def twos_complement(self, num: int) -> str:
        bitstring: str = format((num & 0xFFFFFFFFFFFFFFFF), "064b")
        complement_string: str = ""
        twoC_string: str = ""
        carry_over: int = 0

        for bit in bitstring:
            if bit == "0":
                complement_string += "1"
            else:
                complement_string += "0"

        for i, bit in enumerate(complement_string[:: -1]):
            new_bit: int = 0
            
            if i == 0:
                new_bit, carry_over = self.full_adder(input_a=int(bit), carry_in=1)
                twoC_string += str(new_bit)

            else:
                
                new_bit, carry_over = self.full_adder(input_a=int(bit), carry_in=carry_over)
                twoC_string += str(new_bit)

        twoC_string = twoC_string[:: -1]

        return twoC_string

    
    def is_input_negative(self, input_int: int, bit_len: int = 64) -> bool:
        sign_mask: int = (1 << (bit_len - 1))
        output: bool = False

        if (input_int & sign_mask) != 0:
            output = True

        return  output


    def add_ints(self, int_1: int, int_2:int) -> int:
        i_1: str = format((int_1 & 0xFFFFFFFFFFFFFFFF), "064b")[:: -1] #Flip the bit string for the proper looping
        i_2: str = format((int_2 & 0xFFFFFFFFFFFFFFFF), "064b")[:: -1] #Flip the bit string for the proper looping
        
        carry_over: int = 0
        output: int = 0
        msb_in: int = 0


        for bit_index in range(64):
            new_bit: int = 0
            
            if bit_index == 63:
                msb_in = carry_over

            new_bit, carry_over = self.full_adder(input_a = int(i_1[bit_index]), input_b = int(i_2[bit_index]), carry_in = carry_over)

            mask: int = new_bit << bit_index
            output = output | mask

        if msb_in != carry_over:
            raise AluError(message="add_ints: overflow detected at the end of int addition")

        #Interpret 'result' as a two's‑complement value in 'width' bits, and return it as a Python signed int.
        if self.is_input_negative(input_int = int_1) and self.is_input_negative(input_int = output):
            output = output - (1 << 64) #this will shift back a two's complement integer into the proper python representation

        return output


    def sub_ints(self, int_1: int, int_2:int) -> int:
        i_1: str = format((int_1 & 0xFFFFFFFFFFFFFFFF), "064b")[:: -1] #Flip the bit string for the proper looping
        i_2: str = self.twos_complement(num = int_2)[:: -1]

        carry_over: int = 0
        output: int = 0
        msb_in: int = 0


        for bit_index in range(64):
            new_bit: int = 0

            if bit_index == 63:
                msb_in = carry_over
            
            new_bit, carry_over = self.full_adder(input_a = int(i_1[bit_index]), input_b = int(i_2[bit_index]), carry_in = carry_over)

            mask: int = new_bit << bit_index
            output = output | mask

        if msb_in != carry_over:
            raise AluError(message="sub_ints: overflow detected at the end of int addition")

        #Interpret 'result' as a two's‑complement value in 'width' bits, and return it as a Python signed int.
        if self.is_input_negative(input_int = int_1) and self.is_input_negative(input_int = output):
            output = output - (1 << 64) #this will shift back a two's complement integer into the proper python representation

        return output
            

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

        #Interpret 'result' as a two's‑complement value in 'width' bits, and return it as a Python signed int.
        if (self.is_input_negative(input_int = int_1) or self.is_input_negative(input_int = int_2)) and self.is_input_negative(input_int = output):
            output = output - (1 << 64) #this will shift back a two's complement integer into the proper python representation

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
            raise AluError(message="sub_ints: overflow detected at the end of int addition")
        
        output = self.bit_to_int(new_seq)

        #Interpret 'result' as a two's‑complement value in 'width' bits, and return it as a Python signed int.
        if (self.is_input_negative(input_int = int_1) or self.is_input_negative(input_int = int_2)) and self.is_input_negative(input_int = output):
            output = output - (1 << 64) #this will shift back a two's complement integer into the proper python representation

        return output
    

    #ADD and SUB should have 3 args: dest, operand1, operand2 (tpyes: register, register/int, register/int)
    def add(self, dest: str, operand_1: str | int, operand_2: str | int, op1_type: str, op2_type:str) -> None:
        """Invokes the add_int function from the AU subunit and the read_register_bytes method from the RegisterSupervisor if one of the operands in a register.
        to add the provided operands. The function accepts int and char types, where the value will be the integer representation of a character, to allow
        character transformations.
        
        The function returns none as it directly writes the sum of the operands, into the destination register where the return type is determined
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
            raise AluError(f"iadd: the destination argument must be the name of a valid register, {dest} was given.")

        if isinstance(operand_1, (str)) and op1_type == "register":
            op1_buffer, op1_buffer_type = self.register_supervisor.read_register_bytes(operand_1)
            op1_int_value = self.byte_to_numeric(var = op1_buffer, var_type = "int")[0]

            if op1_buffer_type not in {"int", "char"}:
                raise AluError(f"iadd: int addition can only be performed on int and integer representation of char classes, {op1_buffer_type} was provided.")
        
        elif isinstance(operand_1, (int)) and op1_type == "int":
            op1_int_value = operand_1
            op1_buffer_type = "int"

        else:
            raise AluError(f"iadd: expected type 'register' or 'int' as an operand, {op1_type} was provided.")

        if isinstance(operand_2, (str)) and op2_type == "register":
            op2_buffer, op2_buffer_type = self.register_supervisor.read_register_bytes(operand_2)
            op2_int_value = self.byte_to_numeric(var = op2_buffer, var_type = "int")[0]

            if op2_buffer_type not in {"int", "char"}:
                raise AluError(f"iadd: int addition can only be performed on int and integer representation of char classes, {op2_buffer_type} was provided.")
        
        elif isinstance(operand_2, (int)) and op2_type == "int":
            op2_int_value = operand_2
            op2_buffer_type = "int"

        else:
            raise AluError(f"iadd: expected type 'register' or 'int' as an operand, {op2_type} was provided.")
        
        
        operation_result = self.AU.add_ints(int_1 = op1_int_value, int_2 = op2_int_value, bit_len = 64)
        dest_buffer = self.numeric_to_byte(var = operation_result, var_type = "int")
        dest_type = op1_buffer_type

        self.register_supervisor.write_register(target_register = dest, value = dest_buffer, value_type = dest_type)





#floating point binary single precision - 32 bits
##UNFINISHED and BUGGY

def float_to_binary(num: float, bit_len: int = 32) -> str:
    """
    Convert a floating point number to binary representation in IEEE 754 format.
    
    Args:
        num: Float number to convert
        bit_len: Bit length (32 for single precision, 64 for double precision)
        
    Returns:
        String representation of the binary number
    """
    abs_num: float = abs(num)
    whole_part: int = int(num)
    fraction_part: float = num - whole_part
    sign_bit: str = ""
    output_bin_string: str = ""
    extension_bit_number: int = 5
    rounding_bits: str = ""

    if bit_len == 32:
        exp_len: int = 8
        mant_len: int = 23 
        exp_bias: int = 127
        min_exp: int = -149
        exp_format: str = '08b'

    elif bit_len == 64:
        exp_len: int = 11
        mant_len: int = 52
        exp_bias: int = 1023
        min_exp: int = -1074
        exp_format: str = '011b'
    
    else:
        raise ValueError(f"float_to_binary: Unsupported bit length, 32 or 64 expected, {bit_len} given.")


    if num < 0:
        sign_bit = "1"
    else:
        sign_bit = "0"

    #Handle the 0.0 case
    if num == 0.0:
        return sign_bit + "0" * (bit_len -1)

    #Handle NaN:
    #Take advantage of a unique property of NaN: it's the only value that is not equal to itself. According to IEEE 754:
    #For any normal number x, x == x is always true
    #For NaN, NaN == NaN is always false
    if num != num:
        return sign_bit + ('1' * exp_len) + '1' + ('0' * (mant_len - 1))

    #Handle infinity
    #Take advantage of a fundamental property of IEEE 754 floating-point arithmetic:
    #For any normal number x: x * 0 = 0
    #For infinity: infinity * 0 = NaN (not 0)
    if math.isinf(num):
        return sign_bit + ('1' * exp_len) + ('0' * mant_len)


    wholeP_bin: str = format(abs(whole_part), 'b')
    fractionP_bin: str = ""

    exponent: int = 0
    if abs_num < 2**(1-exp_bias): #for subnormal numbers the exponent field is supposed to be all 0s
        biased_exponent: int = 0 
    
    else:
        exponent: int = math.floor(math.log2(abs_num))
        biased_exponent: int = exponent + exp_bias

    if exponent > exp_bias:
        raise BufferError(f"float_to_binary: overflow detected.")
    elif exponent < min_exp:
        raise BufferError(f"float_to_binary: underflow detected.")
    
    biased_exp_bin: str = format(biased_exponent, exp_format)
        
    fraction_mult: float = abs(fraction_part)
    if abs_num < 2**(1-exp_bias): #fraction for subnormal numbers
        while fraction_mult != 0 and len(fractionP_bin) < (mant_len + extension_bit_number):
            fraction_mult *= 2
            bit: int = int(fraction_mult)
            fractionP_bin += str(bit)
            fraction_mult -= bit
    
    else: #fraction for normal numbers
        while fraction_mult != 0 and ((len(wholeP_bin) - 1) + len(fractionP_bin)) < (mant_len + extension_bit_number):
            fraction_mult *= 2
            bit: int = int(fraction_mult)
            fractionP_bin += str(bit)
            fraction_mult -= bit

    if fraction_part > 0 and fractionP_bin.find("1") == -1:
        raise BufferError(f"float_to_binary: the given number {num} cannot be represented with single precision")

    if abs_num < 2**(1-exp_bias):
        mantissa: str = fractionP_bin.ljust(mant_len, '0')
    
    else:
        mantissa: str = wholeP_bin[1:] + fractionP_bin

    if len(mantissa) < mant_len:
        padding: str = "".join("0" for _ in range(mant_len - len(mantissa)))
        mantissa += padding
    
    elif len(mantissa) > mant_len:
        rounding_bits = mantissa[mant_len:]
        mantissa = mantissa[0 : mant_len]

        biased_exp_bin, mantissa = float_rounder(exponent = biased_exp_bin, mantissa = mantissa, rounding_bits= rounding_bits)
        

    output_bin_string = sign_bit + biased_exp_bin + mantissa

    if len(output_bin_string) != bit_len:
       raise BufferError(f"float_to_binary: the output bit len was expected to be {bit_len}, {len(output_bin_string)} was received")
   

    return output_bin_string



def float_to_binary(num: float, bit_len: int = 32) -> str:
    """
    Convert a floating point number to binary representation in IEEE 754 format.
    
    Args:
        num: Float number to convert
        bit_len: Bit length (32 for single precision, 64 for double precision)
        
    Returns:
        String representation of the binary number
    """
    abs_num: float = abs(num)
    whole_part: int = int(num)
    fraction_part: float = num - whole_part
    sign_bit: str = ""
    output_bin_string: str = ""

    if bit_len == 32:
        exp_len: int = 8
        mant_len: int = 23
        exp_bias: int = 127
        min_exp: int = -149
        exp_format: str = '08b'

    elif bit_len == 64:
        exp_len: int = 11
        mant_len: int = 52
        exp_bias: int = 1023
        min_exp: int = -1074
        exp_format: str = '011b'
    
    else:
        raise ValueError(f"float_to_binary: Unsupported bit length, 32 or 64 expected, {bit_len} given.")


    if num < 0:
        sign_bit = "1"
    else:
        sign_bit = "0"

    #Handle the 0.0 case
    if num == 0.0:
        return sign_bit + "0" * (bit_len -1)

    #Handle NaN:
    #Take advantage of a unique property of NaN: it's the only value that is not equal to itself. According to IEEE 754:
    #For any normal number x, x == x is always true
    #For NaN, NaN == NaN is always false
    if num != num:
        return sign_bit + ('1' * exp_len) + '1' + ('0' * (mant_len - 1))

    #Handle infinity
    #Take advantage of a fundamental property of IEEE 754 floating-point arithmetic:
    #For any normal number x: x * 0 = 0
    #For infinity: infinity * 0 = NaN (not 0)
    if math.isinf(num):
        return sign_bit + ('1' * exp_len) + ('0' * mant_len)


    wholeP_bin: str = format(abs(whole_part), 'b')
    fractionP_bin: str = ""

    exponent: int = 0
    if abs_num < 2**(1-exp_bias): #for subnormal numbers the exponent field is supposed to be all 0s
        biased_exponent: int = 0 
    
    else:
        exponent: int = math.floor(math.log2(abs_num))
        biased_exponent: int = exponent + exp_bias

    if exponent > exp_bias:
        raise BufferError(f"float_to_binary: overflow detected.")
    elif exponent < min_exp:
        raise BufferError(f"float_to_binary: underflow detected.")
    
    biased_exp_bin: str = format(biased_exponent, exp_format)
        
    fraction_mult: float = abs(fraction_part)
    if abs_num < 2**(1-exp_bias): #fraction for subnormal numbers
        while fraction_mult != 0 and len(fractionP_bin) < mant_len:
            fraction_mult *= 2
            bit: int = int(fraction_mult)
            fractionP_bin += str(bit)
            fraction_mult -= bit
    
    else: #fraction for normal numbers
        while fraction_mult != 0 and ((len(wholeP_bin) - 1) + len(fractionP_bin)) < mant_len:
            fraction_mult *= 2
            bit: int = int(fraction_mult)
            fractionP_bin += str(bit)
            fraction_mult -= bit

    if fraction_part > 0 and fractionP_bin.find("1") == -1:
        raise BufferError(f"float_to_binary: the given number {num} cannot be represented with single precision")

    if abs_num < 2**(1-exp_bias):
        mantissa: str = fractionP_bin.ljust(mant_len, '0')
    
    else:
        mantissa: str = wholeP_bin[1:] + fractionP_bin

    if len(mantissa) < mant_len:
        padding: str = "".join("0" for _ in range(mant_len - len(mantissa)))
        mantissa += padding

    output_bin_string = sign_bit + biased_exp_bin + mantissa

    if len(output_bin_string) != bit_len:
       raise BufferError(f"float_to_binary: the output bit len was expected to be {bit_len}, {len(output_bin_string)} was received")
   

    return output_bin_string



#Refactored, working version
def float_to_binary(num: float | int, bit_len: int = 64) -> str:
    """
    Convert a floating point number to binary representation in IEEE 754 format.
    
    Implements complete IEEE 754 standard including:
    - Proper handling of normal and subnormal numbers
    - Special values: +/-0, +/-infinity, NaN  
    - Hardware-accurate rounding with guard/round/sticky bits
    - Overflow and underflow detection
    - Support for both single (32-bit) and double (64-bit) precision
    
    Args:
        num: Float number to convert
        bit_len: Bit length (32 for single precision, 64 for double precision)
        
    Returns:
        String representation of the binary number in IEEE 754 format
        
    Raises:
        ValueError: If bit_len is not 32 or 64
        BufferError: If overflow, underflow, or output length mismatch occurs
    """
    
    #Declare the sign bit variable
    sign_bit: str = ""
    

    #Handle the 32 or 64 bit numbers
    if bit_len == 32:
        exp_len: int = 8
        mant_len: int = 23 
        exp_bias: int = 127
        min_exp: int = -149
        exp_format: str = '08b'

    elif bit_len == 64:
        exp_len: int = 11
        mant_len: int = 52
        exp_bias: int = 1023
        min_exp: int = -1074
        exp_format: str = '011b'
    
    else:
        raise ValueError(f"float_to_binary: Unsupported bit length, 32 or 64 expected, {bit_len} given.")

    #Handle the main edge cases early
    #Handle the sign bit
    if num < 0:
        sign_bit = "1"
    else:
        sign_bit = "0"

    #Handle the 0.0 case
    if num == 0.0:
        return sign_bit + "0" * (bit_len -1)

    #Handle NaN:
    #Take advantage of a unique property of NaN: it's the only value that is not equal to itself. According to IEEE 754:
    #For any normal number x, x == x is always true
    #For NaN, NaN == NaN is always false
    if num != num:
        return sign_bit + ('1' * exp_len) + '1' + ('0' * (mant_len - 1))

    #Handle infinity
    #I choose to use the test from the math library
    if math.isinf(num):
        return sign_bit + ('1' * exp_len) + ('0' * mant_len)
    
    #Declare main common variables
    abs_num: float = abs(num)
    whole_part: int = int(num)
    output_bin_string: str = ""
    extension_bit_number: int = 5
    rounding_bits: str = ""

    #Separate normal and subnormal range handling for building the exponent and mantissa
    if abs_num < 2**(1-exp_bias): #subnormal numbers
        subnorm_bin: str = ""
        
        #exponent for subnormal numbers
        biased_exponent: int = 0 #for subnormal numbers the exponent field is supposed to be all 0s

        #fraction for subnormal numbers
        # For subnormals the stored value is: abs_num = (0.b1b2…bₘ)₂ × 2^(1−bias)
        # so to extract the pure fraction bits 0.b1b2… we divide out 2^(1−bias),
        # i.e. multiply by 2^(bias−1).  The binary “.xxxxx…” expansion of that
        # product yields exactly the mantissa bits for the subnormal case.
        scaled_fraction: float = abs_num * (2 ** (exp_bias - 1))
        while scaled_fraction != 0 and len(subnorm_bin) < (mant_len + extension_bit_number):
            scaled_fraction *= 2
            bit: int = int(scaled_fraction)
            subnorm_bin += str(bit)
            scaled_fraction -= bit

        #mantissa for subnormal numbers
        mantissa: str = subnorm_bin.ljust(mant_len, '0')

    else: #normal numbers
        wholeP_bin: str = format(abs(whole_part), 'b')
        fractionP_bin: str = ""

        #exponent for normal numbers
        exponent: int = math.floor(math.log2(abs_num))
        biased_exponent: int = exponent + exp_bias

        if exponent > exp_bias:
            raise BufferError(f"float_to_binary: overflow detected.")
        elif exponent < min_exp:
            raise BufferError(f"float_to_binary: underflow detected.")
     
        #fraction for normal numbers - formula for the mantissa number = 1.mantissa × 2^(exponent) -> 1.mantissa = number / 2^(exponent)
        norm_farction: float = abs_num / (2 ** exponent) #Normalize the whole number so the mantissa (1.xxx) can be extracted 
        fraction_mult: float = norm_farction - 1 #get the fractional part of the normalized mantissa
        while fraction_mult != 0 and len(fractionP_bin) < (mant_len + extension_bit_number):
            fraction_mult *= 2
            bit: int = int(fraction_mult)
            fractionP_bin += str(bit)
            fraction_mult -= bit

        #mantissa for normal numbers
        mantissa: str = fractionP_bin.ljust(mant_len, '0')
    
    #Translate the biased exponent to binary
    biased_exp_bin: str = format(biased_exponent, exp_format)
    
    #Pad mantissa if needed or round as necessary
    if len(mantissa) < mant_len:
        padding: str = "".join("0" for _ in range(mant_len - len(mantissa)))
        mantissa += padding
    
    elif len(mantissa) > mant_len:
        rounding_bits = mantissa[mant_len:]
        mantissa = mantissa[0 : mant_len]

        biased_exp_bin, mantissa = float_rounder(exponent = biased_exp_bin, mantissa = mantissa, rounding_bits = rounding_bits)
        
    #Build the final output binary and check the output length
    output_bin_string = sign_bit + biased_exp_bin + mantissa

    if len(output_bin_string) != bit_len:
       raise BufferError(f"float_to_binary: the output bit len was expected to be {bit_len}, {len(output_bin_string)} was received")
   

    return output_bin_string
    



float_bin: str = "01000001100011010000000000000000"
import math
from Custom_errors import FpuError

def binary_to_float(fpn_bit_string: str, bit_len: int = 64) -> float:
    """
    Convert IEEE 754 binary representation back to a floating point number.
    
    Handles all IEEE 754 cases including:
    - Normal numbers with implicit leading 1
    - Subnormal numbers (denormalized) 
    - Special values: +/-0, +/-infinity, NaN
    
    Args:
        fpn_bit_string: Binary string representation of the float
        bit_len: Bit length (32 for single precision, 64 for double precision)
        
    Returns:
        Reconstructed floating point number
        
    Raises:
        ValueError: If bit_len is not 32 or 64
    """
    
    sing_bit_string: str = ""
    exponent_string: str = ""
    mantissa_string: str = ""
    output_num: float = 0
    
    sing_bit_string = fpn_bit_string[0]
    if bit_len == 32:
        exponent_string = fpn_bit_string[1:9]
        mantissa_string = fpn_bit_string[9:len(fpn_bit_string)]
        exp_offset: int = 127 #ensures that one can have fractional numbers by shifting the exponent into negative. Roughly half of the range is the bias/offset.
    
    elif bit_len == 64:
        exponent_string = fpn_bit_string[1:12]
        mantissa_string = fpn_bit_string[12:len(fpn_bit_string)]
        exp_offset: int = 1023
    
    else:
        raise ValueError(f"binary_to_float: Unsupported bit length, 32 or 64 expected, {bit_len} given.")


    sing_bit: int = int(sing_bit_string)
    
    mantissa: float = float()
    for bit_index, bit in enumerate(mantissa_string):
        mantissa += int(bit) * (2 ** (-bit_index - 1))
    
    exponent: int = int('0b' + exponent_string, 2)

    #Calculations with special cases
    if exponent == 0 and mantissa == 0: #the case of a 0
        output_num = 0.0
    elif exponent == 0 and mantissa != 0: #the case of very small numbers (subnormal numbers)
        output_num = (-1) ** sing_bit * (0 + mantissa) * (2 ** (1 - exp_offset))
    elif all(bit == '1' for bit in exponent_string) and mantissa == 0: #the case of a too large number = overflows
        if sing_bit == 0:
            return float('Inf')
        elif sing_bit == 1:
            return float('-Inf')
    elif all(bit == '1' for bit in exponent_string) and mantissa != 0: #the case of results of invalid or undefined mathematical operations
        return float('NaN')
    else: # Normal case
        output_num = (-1) ** sing_bit * (1 + mantissa) * (2 ** (exponent - exp_offset))

    return output_num


def full_adder(input_a: int = 0, input_b: int = 0, carry_in: int = 0) -> tuple[int, int]:
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


def float_rounder(exponent: str, mantissa: str, rounding_bits: str) -> tuple[str, str]:
    """
    IEEE 754 compliant rounding using Guard, Round, and Sticky bits.
    
    Implements "round to nearest, ties to even" (banker's rounding):
    - If guard bit = 0: truncate (no rounding)
    - If guard bit = 1:
        - If round = 0 and sticky = 0: round to even (check LSB of mantissa)
        - If round = 1 or sticky = 1: always round up
    
    The rounding may cause mantissa overflow, requiring exponent increment.
    Uses hardware-level full adder for bit-accurate arithmetic operations.
    
    Args:
        exponent: Binary string of the biased exponent
        mantissa: Binary string of the mantissa (without implicit leading bit)
        rounding_bits: Extra precision bits beyond mantissa (guard + round + sticky bits)
        
    Returns:
        tuple[str, str]: (rounded_exponent, rounded_mantissa)
    """
    
    mantissa_lst: list[int] = [int(bit) for bit in mantissa]
    mantissa_lst.reverse() #reversed for bitwise operations!

    exponent_lst: list[int] = [int(bit) for bit in exponent]
    exponent_lst.reverse() #reversed for bitwise operations!

    sticky_bits_lst: list[int] = [int(bit) for bit in rounding_bits[2:]]
    guard_bit: int = int(rounding_bits[0])
    rounding_bit: int = int(rounding_bits[1])
    
    sticky_bit: int = 0
    for bit in sticky_bits_lst:
        sticky_bit = sticky_bit | int(bit)
    
    carry_over: int = 0

    rounded_mantissa: list[int] = []
    rounded_exponent: list[int] = []
    
    rounded_exponent_out: str = ""
    rounded_mantissa_out: str = ""

    if guard_bit == 0: #guard bit (g)
        return exponent, mantissa
    
    elif guard_bit == 1 and rounding_bit == 0 and sticky_bit == 0:
        if mantissa_lst[0] == 0:
            return exponent, mantissa
        
        else:
            for bit_index, bit in enumerate(mantissa_lst):
                new_bit: int = 0

                if bit_index == 0:
                    new_bit, carry_over = full_adder(input_a = bit, carry_in = 1)
                    rounded_mantissa.append(new_bit)
                    
                else:
                    new_bit, carry_over = full_adder(input_a = bit, carry_in = carry_over)
                    rounded_mantissa.append(new_bit)

            if carry_over == 0:
                rounded_mantissa.reverse()

                rounded_mantissa_out = ''.join(str(bit) for bit in rounded_mantissa)
                return exponent, rounded_mantissa_out

            else:
                for bit_index, bit in enumerate(exponent_lst):
                    new_bit: int = 0

                    if bit_index == 0:
                        new_bit, carry_over = full_adder(input_a = bit, carry_in = 1)
                        rounded_exponent.append(new_bit)
                        
                    else:
                        new_bit, carry_over = full_adder(input_a = bit, carry_in = carry_over)
                        rounded_exponent.append(new_bit)

                rounded_exponent.reverse()
                rounded_mantissa.reverse()

                rounded_exponent_out = ''.join(str(bit) for bit in rounded_exponent)
                rounded_mantissa_out = ''.join(str(bit) for bit in rounded_mantissa)

                return rounded_exponent_out, rounded_mantissa_out

    elif guard_bit == 1 and (rounding_bit == 1 or sticky_bit == 1):
        for bit_index, bit in enumerate(mantissa_lst):
            new_bit: int = 0

            if bit_index == 0:
                new_bit, carry_over = full_adder(input_a = bit, carry_in = 1)
                rounded_mantissa.append(new_bit)
                    
            else:
                new_bit, carry_over = full_adder(input_a = bit, carry_in = carry_over)
                rounded_mantissa.append(new_bit)
        
        if carry_over == 0:
            rounded_mantissa.reverse()
            
            rounded_mantissa_out = ''.join(str(bit) for bit in rounded_mantissa)
            return exponent, rounded_mantissa_out

        else:
            for bit_index, bit in enumerate(exponent_lst):
                new_bit: int = 0

                if bit_index == 0:
                    new_bit, carry_over = full_adder(input_a = bit, carry_in = 1)
                    rounded_exponent.append(new_bit)
                            
                else:
                    new_bit, carry_over = full_adder(input_a = bit, carry_in = carry_over)
                    rounded_exponent.append(new_bit)

            rounded_exponent.reverse()
            rounded_mantissa.reverse()

            rounded_exponent_out = ''.join(str(bit) for bit in rounded_exponent)
            rounded_mantissa_out = ''.join(str(bit) for bit in rounded_mantissa)

            return rounded_exponent_out, rounded_mantissa_out

    else:
        return '', ''         


#Refactored, working version
def float_to_binary(num: float | int, bit_len: int = 64) -> str:
    """
    Convert a floating point number to binary representation in IEEE 754 format.
    
    Implements complete IEEE 754 standard including:
    - Proper handling of normal and subnormal numbers
    - Special values: +/-0, +/-infinity, NaN  
    - Hardware-accurate rounding with guard/round/sticky bits
    - Overflow and underflow detection
    - Support for both single (32-bit) and double (64-bit) precision
    
    Args:
        num: Float number to convert
        bit_len: Bit length (32 for single precision, 64 for double precision)
        
    Returns:
        String representation of the binary number in IEEE 754 format
        
    Raises:
        ValueError: If bit_len is not 32 or 64
        BufferError: If overflow, underflow, or output length mismatch occurs
    """
    
    #Declare the sign bit variable
    sign_bit: str = ""
    

    #Handle the 32 or 64 bit numbers
    if bit_len == 32:
        exp_len: int = 8
        mant_len: int = 23 
        exp_bias: int = 127
        min_exp: int = -149
        exp_format: str = '08b'

    elif bit_len == 64:
        exp_len: int = 11
        mant_len: int = 52
        exp_bias: int = 1023
        min_exp: int = -1074
        exp_format: str = '011b'
    
    else:
        raise ValueError(f"float_to_binary: Unsupported bit length, 32 or 64 expected, {bit_len} given.")

    #Handle the main edge cases early
    #Handle the sign bit
    if num < 0:
        sign_bit = "1"
    else:
        sign_bit = "0"

    #Handle the 0.0 case
    if num == 0.0:
        return sign_bit + "0" * (bit_len -1)

    #Handle NaN:
    #Take advantage of a unique property of NaN: it's the only value that is not equal to itself. According to IEEE 754:
    #For any normal number x, x == x is always true
    #For NaN, NaN == NaN is always false
    if num != num:
        return sign_bit + ('1' * exp_len) + '1' + ('0' * (mant_len - 1))

    #Handle infinity
    #I choose to use the test from the math library
    if math.isinf(num):
        return sign_bit + ('1' * exp_len) + ('0' * mant_len)
    
    #Declare main common variables
    abs_num: float = abs(num)
    whole_part: int = int(num)
    output_bin_string: str = ""
    extension_bit_number: int = 5
    rounding_bits: str = ""

    #Separate normal and subnormal range handling for building the exponent and mantissa
    if abs_num < 2**(1-exp_bias): #subnormal numbers
        subnorm_bin: str = ""
        
        #exponent for subnormal numbers
        biased_exponent: int = 0 #for subnormal numbers the exponent field is supposed to be all 0s

        #fraction for subnormal numbers
        # For subnormals the stored value is: abs_num = (0.b1b2…bₘ)₂ × 2^(1−bias)
        # so to extract the pure fraction bits 0.b1b2… we divide out 2^(1−bias),
        # i.e. multiply by 2^(bias−1).  The binary “.xxxxx…” expansion of that
        # product yields exactly the mantissa bits for the subnormal case.
        scaled_fraction: float = abs_num * (2 ** (exp_bias - 1))
        while scaled_fraction != 0 and len(subnorm_bin) < (mant_len + extension_bit_number):
            scaled_fraction *= 2
            bit: int = int(scaled_fraction)
            subnorm_bin += str(bit)
            scaled_fraction -= bit

        #mantissa for subnormal numbers
        mantissa: str = subnorm_bin.ljust(mant_len, '0')

    else: #normal numbers
        wholeP_bin: str = format(abs(whole_part), 'b')
        fractionP_bin: str = ""

        #exponent for normal numbers
        exponent: int = math.floor(math.log2(abs_num))
        biased_exponent: int = exponent + exp_bias

        if exponent > exp_bias:
            raise BufferError(f"float_to_binary: overflow detected.")
        elif exponent < min_exp:
            raise BufferError(f"float_to_binary: underflow detected.")
     
        #fraction for normal numbers - formula for the mantissa number = 1.mantissa × 2^(exponent) -> 1.mantissa = number / 2^(exponent)
        norm_farction: float = abs_num / (2 ** exponent) #Normalize the whole number so the mantissa (1.xxx) can be extracted 
        fraction_mult: float = norm_farction - 1 #get the fractional part of the normalized mantissa
        while fraction_mult != 0 and len(fractionP_bin) < (mant_len + extension_bit_number):
            fraction_mult *= 2
            bit: int = int(fraction_mult)
            fractionP_bin += str(bit)
            fraction_mult -= bit

        #mantissa for normal numbers (already normalized so no need to drop a leading 1)
        mantissa: str = fractionP_bin.ljust(mant_len, '0')
    
    #Translate the biased exponent to binary
    biased_exp_bin: str = format(biased_exponent, exp_format)
    
    #Pad mantissa if needed or round as necessary
    if len(mantissa) < mant_len:
        padding: str = "".join("0" for _ in range(mant_len - len(mantissa)))
        mantissa += padding
    
    elif len(mantissa) > mant_len:
        rounding_bits = mantissa[mant_len:]
        mantissa = mantissa[0 : mant_len]

        biased_exp_bin, mantissa = float_rounder(exponent = biased_exp_bin, mantissa = mantissa, rounding_bits = rounding_bits)
        
    #Build the final output binary and check the output length
    output_bin_string = sign_bit + biased_exp_bin + mantissa

    if len(output_bin_string) != bit_len:
       raise BufferError(f"float_to_binary: the output bit len was expected to be {bit_len}, {len(output_bin_string)} was received")
   

    return output_bin_string


def int_to_bits(input_int: int, bit_len: int = 64) -> list[int]:
        """
        Convert an integer to its binary representation as a list of bits.
    
        Args:
            input_int: The integer to convert
            bit_len: The bit width to use (default: 64)
        
        Returns:
            A list of bits, with least significant bit first
        """
        
        width_mask: int = (1 << bit_len) -1 #creates a mask of 64 1s to cut the target to 64 bits
        bit_seq: list[int] = []

        masked_input = input_int & width_mask

        for bit_index in range(bit_len):
            bit_seq.append((masked_input >> bit_index) & 1)

        return bit_seq

def bit_to_int(input_bits: list[int]) -> int:
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

def fp_twos_complement(bit_seq: list[int]) -> list[int]:
        """
        Converts a bit sequence to it's two's complement representation.
    
        Args:
            num: An integer to convert
        
        Returns:
            A two's complement bit list representation of the given number with least significant bit first.
        """

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
                new_bit, carry_over = full_adder(input_a = bit, carry_in = 1)
                twoC_seq.append(new_bit)

            else:
                
                new_bit, carry_over = full_adder(input_a = bit, carry_in = carry_over)
                twoC_seq.append(new_bit)

        return twoC_seq

def add_biased_exponents(exponent_1: list[int], exponent_2: list[int], intermediate_len: int) -> list[int]:
    """
    Add two biased IEEE 754 exponents for floating-point multiplication.
    
    Performs binary addition of two exponent bit sequences with overflow detection.
    The exponents are added in their biased form (no bias removal required).
    Used as part of floating-point multiplication: new_exp = exp1 + exp2 - bias.
    
    Args:
        exponent_1: First exponent as list of bits in MSB->LSB order
        exponent_2: Second exponent as list of bits in MSB->LSB order  
        intermediate_len: Target bit length for intermediate calculations (includes overflow protection)
        
    Returns:
        Sum of exponents as list of bits in LSB->MSB order, padded to intermediate_len
        
    Raises:
        AluError: If arithmetic overflow is detected during addition
        
    Notes:
        - Input exponents are automatically zero-padded to intermediate_len
        - Uses full_adder for bit-by-bit addition with carry propagation
        - Output is in LSB->MSB order for further processing
    """

    #Borrow the exponents
    exp_1: list[int] = exponent_1.copy()
    exp_2: list[int] = exponent_2.copy()

    #Reverse the exponents for easier bitwise operations
    exp_1.reverse()
    exp_2.reverse()
        
    #Declare variables
    carry_over: int = 0
    new_seq: list[int] = []
    msb_in: int = 0

    #Pad the exponents to the intermediate bit length 
    if len(exp_1) != intermediate_len or len(exp_2) != intermediate_len:
        exp_1.extend([0 for _ in range(intermediate_len - len(exp_1))])
        exp_2.extend([0 for _ in range(intermediate_len - len(exp_2))])

    #Do bitwise addition with the full adder
    for bit_index in range(intermediate_len):
        new_bit: int = 0
            
        if bit_index == (intermediate_len - 1):
            msb_in = carry_over

        new_bit, carry_over = full_adder(input_a = exp_1[bit_index], input_b = exp_2[bit_index], carry_in = carry_over)

        new_seq.append(new_bit)

    #Check for overflow
    if msb_in != carry_over:
        raise FpuError(message="add_ints: overflow detected at the end of int addition")

    return new_seq
    

def sub_bias(exponent_seq: list[int], bias: int, intermediate_len: int, final_len: int, subnormal: bool) -> list[int]:
    """
    Subtract the IEEE 754 exponent bias from a biased exponent sum.
    
    Removes one bias from the exponent sum (exp1 + exp2 - bias) to get the final
    exponent for floating-point multiplication. Uses two's complement arithmetic
    for subtraction with overflow detection.
    
    Args:
        exponent_seq: Biased exponent sum as list of bits in LSB->MSB order
        bias: IEEE 754 bias value (127 for 32-bit, 1023 for 64-bit)
        intermediate_len: Bit length of input exponent sequence
        final_len: Target bit length for final exponent output
        
    Returns:
        Unbiased exponent as list of bits in MSB->LSB order, trimmed to final_len
        
    Raises:
        AluError: If arithmetic overflow is detected during subtraction
        
    Notes:
        - Converts bias to two's complement for subtraction
        - Trims result from intermediate_len to final_len bits
        - Reverses output to MSB->LSB order for IEEE 754 format
    """
    ###NOTE: PROBLEM HERE, LOOK INTO THIS PART
    if subnormal == True: #for subnormal numbers we use the Ex + Ey (done before) - (1-[127 + nlz]) formula to get a normalized subnormal exponent
        bias_seq: list[int] = int_to_bits(input_int = (1 - bias), bit_len = intermediate_len)
        bias_2c: list[int] = bias_seq #as the generated bias is negative, it is already in two's complement form after translation into a bit seq
        print(f"Here is the bias {1 - bias} and the bias in 2c form {bias_2c}")
    else:
        bias_seq: list[int] = int_to_bits(input_int = bias, bit_len = intermediate_len)
        bias_2c: list[int] = fp_twos_complement(bit_seq = bias_seq)

    carry_over: int = 0
    new_seq: list[int] = []
    msb_in: int = 0


    for bit_index in range(intermediate_len):
        new_bit: int = 0

        if bit_index == (intermediate_len - 1):
            msb_in = carry_over
            
        new_bit, carry_over = full_adder(input_a = exponent_seq[bit_index], input_b = bias_2c[bit_index], carry_in = carry_over)

        new_seq.append(new_bit)

    if msb_in != carry_over:
        raise FpuError(message="sub_bias: overflow detected at the end of bias subtraction")

    print(f"This is the new sequence after sum: {new_seq}")

     #detect a subnormal result in case of a subnormal input which should produce an all 0 exponent by looking at the MSB of the intermediate result (-512/-4096)
    if subnormal == True and new_seq[-1] == 1:
        output: list[int] = [0 for _ in range(final_len)] #Subnormal exponent pattern: all 1s
        print(f"This is the new sequence after sum: {new_seq} so this branch should be active")
    
    #detect overflow which should produce an Inf value
    elif subnormal == False and new_seq[final_len : len(new_seq)].count(1) != 0: #there are 1s over the exponent bit limit
        output: list[int] = [1 for _ in range(final_len)] #Inf exponent pattern: all 0s
        print(f"This is the new sequence after sum: {new_seq} so this branch should NOT be active")
    
    else:
        output: list[int] = new_seq[0:final_len]
        print(f"This is the new sequence after sum: {new_seq} so this branch NOT should be active")
    
    output = output[::-1]

    return output

def mant_multiplier(mantissa_1: list[int], mantissa_2: list[int], new_exponent: list[int], mantissa_length: int,
                     subn_multiplicand: bool, subn_multiplier: bool, nlz_multiplicand: int, nlz_multiplier: int) -> tuple[list[int], list[int]]:
    """
    Multiply two IEEE 754 mantissas with hidden bit restoration and overflow handling.
    
    Performs binary multiplication of two normalized mantissas using shift-and-add
    algorithm. Handles mantissa overflow by incrementing exponent and normalizing
    the result. Implements proper hidden bit management for IEEE 754 compliance.
    
    The multiplication follows the pattern:
    1. Restore hidden bits (1.xxx format)
    2. Multiply using partial products with left shifts
    3. Sum all partial products
    4. Check for overflow and normalize if needed
    5. Extract final mantissa (fractional part only)
    
    Args:
        mantissa_1: First mantissa (multiplicand) as list of fractional bits in MSB->LSB order
        mantissa_2: Second mantissa (multiplier) as list of fractional bits in MSB->LSB order
        new_exponent: Current exponent sum as list of bits in MSB->LSB order
        mantissa_length: Length of input mantissas (23 for 32-bit, 52 for 64-bit)
        
    Returns:
        Tuple containing:
        - Updated exponent as list of bits in MSB->LSB order (incremented if overflow)
        - Final mantissa as list of fractional bits (hidden bit removed)
        
    Raises:
        AluError: If overflow is detected during partial product addition
        
    Notes:
        - Automatically restores hidden bits before multiplication
        - Uses overflow guard bit at MSB to detect normalization need
        - Result mantissa excludes both overflow guard bit and hidden bit
        - Supports variable precision (determined by mantissa_length parameter)
        
    Algorithm Details:
        - Generates partial products using shift-and-add method
        - Result length is 2 * (mantissa_length + 1) bits
        - Overflow detection checks MSB of final product
        - Normalization: right shift + exponent increment if MSB = 1
    """

    #Declare new variables
    intermed_product: list[int] = []
    product_lst: list[list[int]] = []
    product_sum: list[int] = []
    
    #Borrow the mantissas and the exponent
    mant_1: list[int] = mantissa_1.copy() #multiplicand
    mant_2: list[int] = mantissa_2.copy() #multiplier
    exp: list[int] = new_exponent.copy()

    #Check if there is a subnormal number and re-insert the hidden 0 or 1 into the mantissa sequences accordingly
    if subn_multiplicand == True and subn_multiplier == True:
        mant_1.insert(0, 0) #subnormal, re-insert a 0
        mant_2.insert(0, 0) #subnormal, re-insert a 0

        #remove the leading 0s to normalize the mantissas (1st step of a left shift)
        mant_1 = mant_1[nlz_multiplicand : len(mant_1)]
        mant_2 = mant_2[nlz_multiplier : len(mant_2)]

        #pad them to normal length (2nd step ofa left shift)
        [mant_1.append(0) for _ in range(nlz_multiplicand)]
        [mant_2.append(0) for _ in range(nlz_multiplier)]
    
    elif subn_multiplicand == True and subn_multiplier == False:
        mant_1.insert(0, 0) #subnormal, re-insert a 0
        mant_2.insert(0, 1) #normal, re-insert a 1

        #remove the leading 0s to normalize the mantissas (1st step of a left shift)
        mant_1 = mant_1[nlz_multiplicand : len(mant_1)]

        #pad them to normal length (2nd step ofa left shift)
        [mant_1.append(0) for _ in range(nlz_multiplicand)]

    elif subn_multiplier == True and subn_multiplicand == False:
        mant_1.insert(0, 1) #normal, re-insert a 1
        mant_2.insert(0, 0) #subnormal, re-insert a 0

        #remove the leading 0s to normalize the mantissas (1st step of a left shift)
        mant_2 = mant_2[nlz_multiplier : len(mant_2)]

        #pad them to normal length (2nd step ofa left shift)
        [mant_2.append(0) for _ in range(nlz_multiplier)]

    else:
        mant_1.insert(0, 1) #normal, re-insert a 1
        mant_2.insert(0, 1) #normal, re-insert a 1
    
    #Reverse the bit order of the multiplier for easier calculations (leave the multiplicand as is)
    mant_2.reverse()

    #Multiply the mantissa sequences and store the intermediate products
    for position, bit in enumerate(mant_2):
        intermed_product = []

        
        for b in mant_1: #multiplication loop
            intermed_product.append(bit * b) #bit = multiplier bit, b = multiplicand bit

        intermed_product.extend([0] * position) #left shift
        product_lst.append(intermed_product) #store intermed products

    #Final product length (twice the mantissa with the hidden bit)
    final_product_len: int = 2 * (mantissa_length + 1)

    #Pad all intermediate products to full length for the addition
    for ip in product_lst:
        if len(ip) != final_product_len:
            while len(ip) != final_product_len:
                ip.insert(0, 0) #can only be done once at a time unlike extend hence the while loop
        
        else:
            continue

    #Reverse all intermed_products for easier addition
    for ip in product_lst:
        ip.reverse()

    #Sum the intermed_products
    for i, ip in enumerate(product_lst):
        if i == 0:
            product_sum = ip

        else:
            carry_over: int = 0
            new_mant_seq: list[int] = []

            for bit_index in range(final_product_len): #add the intermediate_product to the sum
                new_bit: int = 0

                new_bit, carry_over = full_adder(input_a = product_sum[bit_index], input_b = ip[bit_index], carry_in = carry_over)

                new_mant_seq.append(new_bit)

            #Check for overflow between intermediate products
            if carry_over != 0:
                raise AluError(message="mant_multiplier: overflow detected during intermediate product addition")

            product_sum: list[int] = new_mant_seq #update the sum with the additional product
            

    #Reverse the new mantissa bit string back to MSB->LSB
    print(f"This is mantissa product sum before reversing to MSB -> LSB and exp overflow check {product_sum}")
    
    product_sum.reverse()
    
    print(f"This is mantissa product sum before exp overflow check {product_sum}")
    #Carry the final mantissa overflow into the new exponent
    if product_sum[0] == 1: #Checks if there is mantissa overflow into the exponent by looking at the MSB, if it is 1 the mantissa needs normalization /
                            #(the decimal point must float up and the leading 1 is new hidden bit and only that is discarded)
        exp.reverse() #reverse the exponent for easier addition

        new_exp_seq: list[int] = []
        carry_over: int = 0
        msb_in: int = 0

        for bit_index in range(len(exp)):
            new_bit: int = 0

            if bit_index == (len(exp) - 1):
                msb_in = carry_over

            if bit_index == 0:
                new_bit, carry_over = full_adder(input_a = exp[bit_index], input_b = 0, carry_in = 1)

            else:
                new_bit, carry_over = full_adder(input_a = exp[bit_index], input_b = 0, carry_in = carry_over)

            new_exp_seq.append(new_bit)

        #Check for overflow
        if msb_in != carry_over:
            raise AluError(message="mant_multiplier: overflow detected during exponent adjustment")

        #Flip the new exponent bit string back to MSB-LSB
        new_exp_seq.reverse()

        #Remove the leading 1 as a hidden bit from the mantissa
        product_sum: list[int] = product_sum[1:len(product_sum)]

        #Return the new exponent and mantissa bit strings
        return new_exp_seq, product_sum

    else: #if the MSB is 0 there is no overflow, no normalization is needed, however the MSB must be discarded and the next bit is the hidden bit which must be removed
        #Return the original exponent and the new mantissa bit strings
        extracted_mantissa: list[int] = product_sum[2:]  # Skip overflow bit (MSB) and hidden bit
        return exp, extracted_mantissa


def float_multiplier(multiplicand: float | int, multiplier: float | int, precision: int = 64) -> float:
    """
    Multiply two floating-point numbers using IEEE 754 compliant binary arithmetic.
    
    Implements complete IEEE 754 floating-point multiplication including:
    - Exponent addition with bias handling
    - Mantissa multiplication with hidden bit management  
    - Overflow detection and normalization
    - Proper rounding with guard/round/sticky bits
    - Sign bit calculation (XOR of input signs)
    - Support for both single (32-bit) and double (64-bit) precision
    
    The multiplication follows IEEE 754 formula:
    Result = (±1) × (mant1 × mant2) × 2^(exp1 + exp2 - bias)
    
    Args:
        multiplicand: First number to multiply (float or int)
        multiplier: Second number to multiply (float or int)  
        precision: Bit precision (32 for single, 64 for double precision)
        
    Returns:
        Product as IEEE 754 compliant floating-point number
        
    Raises:
        TypeError: If arguments are not float/int or precision is not int
        ValueError: If precision is not 32 or 64
        AluError: If arithmetic overflow occurs during computation
        BufferError: If internal buffer operations fail
        
    Notes:
        - Automatically handles precision-specific parameters (exponent length, bias, etc.)
        - Uses extended precision for intermediate calculations
        - Implements proper IEEE 754 rounding for final result
        - Maintains bit-level accuracy throughout computation
        
    Precision Specifications:
        32-bit: 8-bit exponent, 23-bit mantissa, bias=127
        64-bit: 11-bit exponent, 52-bit mantissa, bias=1023
        
    Algorithm Flow:
        1. Convert inputs to binary representation
        2. Extract and add exponents (with bias subtraction)
        3. Multiply mantissas with overflow handling
        4. Round result using extended precision
        5. Assemble final IEEE 754 bit pattern
        6. Convert back to native float format
    """

    #Argument type checks
    if not isinstance(multiplicand, (float, int)) or not isinstance(multiplier, (float, int)):
        raise TypeError(f"float_multiplier: the arguments num_1 and num_2 must be of type float or int, {isinstance(num_1, (float, int))} and {isinstance(num_2, (float, int))} were provided.")

    if not isinstance(precision, (int)):
        raise TypeError(f"float_multiplier: the argument precision must be of type int, {isinstance(precision, (int))} was provided.")


    #Convert the inputs to bit strings
    num_1_bits: str = float_to_binary(num = multiplicand, bit_len = precision)
    num_2_bits: str = float_to_binary(num = multiplier, bit_len = precision)


    #Convert the bit strings to lists
    n1_bit_lst: list[int] = [int(i) for i in num_1_bits]
    n2_bit_lst: list[int] = [int(i) for i in num_2_bits]


    #Define features based on precision
    if precision == 32:
        exp_len: int = 8
        mant_len: int = 23
        exp_bias: int = 127
        intermediate_buffer_len: int = 10

    elif precision == 64:
        exp_len: int = 11
        mant_len: int = 52
        exp_bias: int = 1023
        intermediate_buffer_len: int = 13

    else:
        raise ValueError(f"float_multiplier: 32 or 64 was expected for precision, {precision} was provided.")
    

    #Weird edge case check for 0 * +/- Inf which should produce a NaN
    if (n1_bit_lst.count(1) == 0 or n2_bit_lst.count(1) == 0) and ((n1_bit_lst[1 : exp_len + 1].count(0) == 0 and n1_bit_lst[exp_len + 1 : mant_len].count(1) == 0)
                                                                   or (n2_bit_lst[1 : exp_len + 1].count(0) == 0 and n2_bit_lst[exp_len + 1 : mant_len].count(1) == 0)):
        final_exponent: str = "1" * exp_len #exponent must be all 1s
        
        final_mantissa: str = "1" + ("0" * (mant_len - 1)) #mantissa must be all 0s with a leading 1

        new_sign_bit = n1_bit_lst[0] ^ n2_bit_lst[0]
        final_sign_bit: str = str(new_sign_bit)

        float_out_bit_string: str = final_sign_bit + final_exponent + final_mantissa

        return binary_to_float(fpn_bit_string = float_out_bit_string, bit_len = precision)
    

    #Normal zero multiplication check and exit upon 0 multiplier or multiplicand
    if n1_bit_lst.count(1) == 0 or n2_bit_lst.count(1) == 0:
        final_exponent: str = "0" * exp_len #exponent must be all 0s
        
        final_mantissa: str = "0" * mant_len #mantissa must be all 0s

        new_sign_bit = n1_bit_lst[0] ^ n2_bit_lst[0]
        final_sign_bit: str = str(new_sign_bit)

        float_out_bit_string: str = final_sign_bit + final_exponent + final_mantissa

        return binary_to_float(fpn_bit_string = float_out_bit_string, bit_len = precision)

    #Input infinity check and exit upon exponent overflow
    if n1_bit_lst[1 : exp_len + 1].count(0) == 0 or n2_bit_lst[1 : exp_len + 1].count(0) == 0: #only 1s, no 0s
        final_exponent: str = ""
        for bit in range(exp_len):
            final_exponent += str(bit)

        final_mantissa: str = "0" * mant_len #mantissa must be all 0s

        new_sign_bit = n1_bit_lst[0] ^ n2_bit_lst[0]
        final_sign_bit: str = str(new_sign_bit)

        float_out_bit_string: str = final_sign_bit + final_exponent + final_mantissa
        print(f"This branch got activated {new_exponent} if the input is inf")

        return binary_to_float(fpn_bit_string = float_out_bit_string, bit_len = precision)
    

    #Separate the exponent and mantissa bits
    num_1_exp: list[int] = n1_bit_lst[1 : exp_len + 1]
    num_2_exp: list[int] = n2_bit_lst[1 : exp_len + 1]

    multiplicand_mantissa: list[int] = n1_bit_lst[exp_len + 1 : (exp_len + 1) + mant_len] #bit 9 -> bit 32 in a 32 bit float (bit 32 is exclusive)
    multiplier_mantissa: list[int] = n2_bit_lst[exp_len + 1 : (exp_len + 1) + mant_len]


    #Check for subnormal numbers and normalize them if present
    subnormal_num: bool = False
    subnormal_multiplicand: bool = False
    subnormal_multiplier: bool = False
    
    nlz_multiplicand: int = 0 #number of leading 0s which is given by the mantissa
    nlz_multiplier: int = 0 #number of leading 0s which is given by the mantissa

    if num_1_exp.count(1) == 0 and (n1_bit_lst[exp_len + 1 : mant_len].count(1) != 0):
        subnormal_multiplicand: bool = True
        subnormal_num: bool = True
        for bit in multiplicand_mantissa:
            if bit == 0:
                nlz_multiplicand += 1
            else:
                break

        nlz_multiplicand += 1 #add the hidden leading 0

    if num_2_exp.count(1) == 0 and (n2_bit_lst[exp_len + 1 : mant_len].count(1) != 0):
        subnormal_multiplier: bool = True
        subnormal_num: bool = True
        for bit in multiplier_mantissa:
            if bit == 0:
                nlz_multiplier += 1
            else:
                break
        
        nlz_multiplier += 1 #add the hidden leading 0
    

    #Calculate the new exponent
    #NOTE: by subtracting one bias I do not need to re-add the bias at the end of the operation to get the proper exponent bit string
    exponent_sum: list[int] = add_biased_exponents(exponent_1 = num_1_exp, exponent_2 = num_2_exp, intermediate_len = intermediate_buffer_len)
    #NOTE: for subnormal numbers the exponent bias will be 1 - (bias (-126 or -1022) + number of leading zeros) which should not effect normal number calculations as /
    #  1-bias is on a different if branch while nlz will be 0 for a normal number
    if subnormal_multiplicand == True and subnormal_multiplier == True:
        new_exponent: list[int] = sub_bias(exponent_seq = exponent_sum, bias = (exp_bias + nlz_multiplicand + nlz_multiplier), intermediate_len = intermediate_buffer_len, final_len = exp_len, subnormal = subnormal_num)
    
    elif subnormal_multiplicand == True and subnormal_multiplier == False:
        new_exponent: list[int] = sub_bias(exponent_seq = exponent_sum, bias = (exp_bias + nlz_multiplicand), intermediate_len = intermediate_buffer_len, final_len = exp_len, subnormal = subnormal_num)
        print(f"This branch should be active\n{subnormal_num}\n{subnormal_multiplicand}\n{subnormal_multiplier}\n{nlz_multiplicand}\n{nlz_multiplier}\n{num_1_exp}\n{num_2_exp}\n{exponent_sum}\n{exp_bias}.")
    elif subnormal_multiplicand == False and subnormal_multiplier == True:
        new_exponent: list[int] = sub_bias(exponent_seq = exponent_sum, bias = (exp_bias + nlz_multiplier), intermediate_len = intermediate_buffer_len, final_len = exp_len, subnormal = subnormal_num)

    else:
        new_exponent: list[int] = sub_bias(exponent_seq = exponent_sum, bias = exp_bias, intermediate_len = intermediate_buffer_len, final_len = exp_len, subnormal = subnormal_num)
        print("This should not activate.")

    
    print(f"Nex exp after summ and sub: {new_exponent}")

    #Calculate the new, full length mantissa product and the potential new exponent
    new_exponent, mantissa_product = mant_multiplier(mantissa_1 = multiplicand_mantissa, mantissa_2 = multiplier_mantissa, new_exponent = new_exponent, mantissa_length = mant_len,
                                                     subn_multiplicand = subnormal_multiplicand, subn_multiplier = subnormal_multiplier, nlz_multiplicand = nlz_multiplicand, 
                                                     nlz_multiplier = nlz_multiplier)

    print(f"Nex exp after mant multipl and mant. product: {new_exponent}\n{mantissa_product}")
    #Round and trim the new mantissa_product to the proper length and handle potential rounding overflow into the new exponent
    new_extended_mantissa: list[int] = mantissa_product #use the full mantissa product as an extended mantissa for rounding

    extended_mantissa_string: str = "" #convert the extended mantissa to a string for the float rounding
    exponent_string: str = "" #convert the extended exponent to a string for the float rounding

    for bit in new_extended_mantissa: #mantissa string conversion
        extended_mantissa_string += str(bit)

    for bit in new_exponent: #exponent string conversion
        exponent_string += str(bit)

    rounding_bits: str = extended_mantissa_string[mant_len : (mant_len + 5)] #prepare the extra bits for rounding
    mantissa_string: str = extended_mantissa_string[0 : mant_len] #prepare the mantissa for rounding
    
    final_exponent, final_mantissa = float_rounder(exponent = exponent_string, mantissa = mantissa_string, rounding_bits = rounding_bits)

    print(f"Final exp and final mant. after rounding: {final_exponent}\n{final_mantissa}")
    #Infinity check for the final exponent value and exit upon exponent overflow
    if final_exponent.rfind("0") == -1: #only 1s, no 0s
        final_exponent: str = ""
        for bit in new_exponent:
            final_exponent += str(bit)

        final_mantissa: str = "0" * mant_len #mantissa must be all 0s

        new_sign_bit = n1_bit_lst[0] ^ n2_bit_lst[0]
        final_sign_bit: str = str(new_sign_bit)

        float_out_bit_string: str = final_sign_bit + final_exponent + final_mantissa
        print(f"This branch got activated {new_exponent} if the final exp is inf")

        return binary_to_float(fpn_bit_string = float_out_bit_string, bit_len = precision)
    

    #Decide the sign bit using the xor operation
    new_sign_bit = n1_bit_lst[0] ^ n2_bit_lst[0]
    final_sign_bit: str = str(new_sign_bit)


    #Assemble the new floating point number as a bit string
    float_out_bit_string: str = final_sign_bit + final_exponent + final_mantissa
    print(float_out_bit_string)


    #Convert the new floating point bit string into a floating point number
    float_out: float = binary_to_float(fpn_bit_string = float_out_bit_string, bit_len = precision)

    return float_out





