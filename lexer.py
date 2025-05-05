from __future__ import annotations


def lexer(instructions: list[str], instruction_set: list[str], register_names: set[str]) -> list["Token"]:
    token_lst: list["Token"] = []


    def operand_classifier(operand: str) -> "Token":
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

from Token import Token