from __future__ import annotations

from Custom_errors import RegisterError


def code_executor(instruction_lst: list["Instruction"], jump_tbl: dict[str, int], register_tbl: "RegisterSupervisor") -> str | int:
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

from Instruction import Instruction
from Registers import Register
from Registers import RegisterSupervisor