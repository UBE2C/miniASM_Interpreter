from __future__ import annotations


def parser(token_list: list["Token"]) -> tuple[list["Instruction"], dict[str, int]]:
    token_lst: list["Token"] = token_list.copy()
    instruction_list: list["Instruction"] = []
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

from Token import Token
from Instruction import Instruction