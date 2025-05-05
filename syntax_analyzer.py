from __future__ import annotations

from Custom_errors import RegisterError


def syntax_analyzer(instruction_lst: list["Instruction"], jump_tbl: dict[str, int], register_tbl: "RegisterSupervisor") -> list[str]:
    error_list: list[str] = []

    for i, instruction in enumerate(instruction_lst):
        if instruction.opcode in ("mov", "add", "sub", "mul", "div", "fdiv"):
            if len(instruction.args) < 2:
                error_list.append(f"Instruction {i}: {instruction.opcode} expects exactly 2 arguments, less was given.")
            
            elif len(instruction.args) > 2:
                error_list.append(f"Instruction {i}: {instruction.opcode} expects exactly 2 arguments, more was given.")

            if not isinstance(instruction.args[0], (str)):
                error_list.append(f"Instruction {i}: {instruction.opcode} expects a register name as a first argument, none str type was given.")

            elif isinstance(instruction.args[0], (str)) and instruction.args[0] not in register_tbl.list_registers():
                error_list.append(f"Instruction {i}: {instruction.opcode} expects a valid register name as a first argument, the given name is not part of the mapped registers.")
            
            if isinstance(instruction.args[1], (str)) and instruction.args[1] not in register_tbl.list_registers():
                error_list.append(f"Instruction {i}: {instruction.opcode} expects an int, float or a valid register name as the second argument, the given name is not part of the mapped registers.")

        elif instruction.opcode in ("inc", "dec"):
            if len(instruction.args) < 1:
                error_list.append(f"Instruction {i}: {instruction.opcode} expects exactly 1 argument, less was given.")
            
            elif len(instruction.args) > 1:
                error_list.append(f"Instruction {i}: {instruction.opcode} expects exactly 1 argument, more was given.")

            if not isinstance(instruction.args[0], (str)):
                error_list.append(f"Instruction {i}: {instruction.opcode} expects a register name as an argument, none str type was given.")

            elif isinstance(instruction.args[0], (str)) and instruction.args[0] not in register_tbl.list_registers():
                error_list.append(f"Instruction {i}: {instruction.opcode} expects a valid register name as an argument, the given name is not part of the mapped registers.")

        elif instruction.opcode in ("jmp", "jne", "je", "jge", "jg", "jle", "jl", "call"):
            if len(instruction.args) < 1:
                error_list.append(f"Instruction {i}: {instruction.opcode} expects exactly 1 argument, less was given.")
            
            elif len(instruction.args) > 1:
                error_list.append(f"Instruction {i}: {instruction.opcode} expects exactly 1 argument, more was given.")

            if not isinstance(instruction.args[0], (str)):
                error_list.append(f"Instruction {i}: {instruction.opcode} expects a label of type str as an argument, none str type was given.")

            elif isinstance(instruction.args[0], (str)) and instruction.args[0] not in jump_tbl.keys():
                error_list.append(f"Instruction {i}: {instruction.opcode} expects a valid label argument, the given label is not part of the mapped labels.")
            
        elif instruction.opcode == "cmp":
            if len(instruction.args) < 2:
                error_list.append(f"Instruction {i}: 'cmp' expects exactly 2 arguments, less was given.")
            
            elif len(instruction.args) > 2:
                error_list.append(f"Instruction {i}: 'cmp' expects exactly 2 arguments, more was given.")

            elif isinstance(instruction.args[0], (str)) and instruction.args[0] not in register_tbl.list_registers():
                error_list.append(f"Instruction {i}: 'cmp' expects an int, float or a valid register name as a first argument, the given name is not part of the mapped registers.")
            
            if isinstance(instruction.args[1], (str)) and instruction.args[1] not in register_tbl.list_registers():
                error_list.append(f"Instruction {i}: 'cmp' expects an int, float or a valid register name as the second argument, the given name is not part of the mapped registers.")

        elif instruction.opcode == "msg":
            for e, arg in enumerate(instruction.args):
                if not isinstance(arg, (str)):
                    error_list.append(f"Instruction {i}: {instruction.opcode} expects a character strings or valid register names arguments, argument {e} is not of type str.")

                elif isinstance(arg, (str)) :
                    if not (arg.startswith("'") and arg.endswith("'")) and arg not in register_tbl.list_registers():
                        error_list.append(f"Instruction {i}: {instruction.opcode} argument {e} is neither a valid register nor a string literal.")

        elif instruction.opcode in ("ret", "end"):
            if len(instruction.args) != 0:
                error_list.append(f"Instruction {i}: {instruction.opcode} expects no argument, but some was given.")

    return error_list

from Instruction import Instruction
from Registers import Register
from Registers import RegisterSupervisor