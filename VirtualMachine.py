#----- Import custom classes -----#
from Token import Token
from Instruction import Instruction
from Registers import Registers
from Memory import Memory

#----- Import custom errors -----#
from Custom_errors import RegisterError
from Custom_errors import vRAMError

#----- Import sub-functions -----#
from lexer import lexer
from parser import parser
from syntax_analyzer import syntax_analyzer
from code_executor import code_executor


class VirtualMachine:
    INSTRUCTION_SET: list[str] = ["mov", "inc", "dec", "add", "sub", "mul", "div", "fdiv", ":", "jmp",
                                 "cmp", "jne", "je", "jge", "jg", "jle", "jl", "call", "ret", "str", "ldr",
                                 "msg", ";", "end"]
    
    def __init__(self, code) -> None:
        self.code: str = code
        self.line_pointer: int = 0
        self.instruction_set: list[str] = self.INSTRUCTION_SET
        self.preprocessed_code = self.code.split("\n")
        self.token_list: list[Token] = []
        self.instruction_list: list[Instruction] = []
        self.jump_table: dict[str, int] = {}
        self.register_table: Registers = Registers(31)
        self.vRAM: Memory = Memory(256)
        self.error_list: list[str] = []
        self.output_stream: str | int = ""
        
        
    def __str__(self) -> str:
        return f"Program:\n< {len(self.preprocessed_code)} lines,\n{len(self.token_list)} tokens,\n{len(self.instruction_list)} instructions,\n{len(self.register_table)} registers >"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    
    def lex(self) -> None:
        self.token_list = lexer(instructions = self.preprocessed_code, instruction_set = self.instruction_set)

    def parse(self) -> None:
        self.instruction_list, self.jump_table = parser(token_list = self.token_list)

    def analyze_syntax(self) -> None:
        self.error_list = syntax_analyzer(instruction_lst = self.instruction_list, jump_tbl = self.jump_table, register_tbl = self.register_table)

    def execute_code(self) -> None:
        self.output_stream = code_executor(instruction_lst = self.instruction_list, jump_tbl = self.jump_table, register_tbl = self.register_table)

    def interpret(self) -> None:
        self.lex()
        self.parse()
        self.map_registers()
        self.analyze_syntax()
        if len(self.error_list) != 0:
            raise SyntaxError(f"Syntax errors found in provided code:\n {self.error_list} ")
        self.execute_code()
        return self.output_stream


"""
Plans:

Structures:
maybe adding the option to assign variable names to RAM addresses (so basically named pointers, I think) so retrieval of strings are more inituative (this would require a separate dict which keeps track of the variable name as key, and the memory address as value)

New instructions: 
alloc - an integer/float/string/register argument. It should move the arg value to the vRAM to the address stored in the link register 
store - should take an integer/float/string/register and an immediate int as an argument. should automatically pass the second int arg to the link register and call alloc to store the first argument
load - takes a register name as an argument. It should move the stored value from the vRAM (based on the address stored in the link register) to the arg register.
retrieve - should take a register and an immediate int as an argument. should automatically pass the second int arg to the link register and call load to retrieve the stored value into the first argument.
(they seem kind of redundant, but allows separate management of the LR. Maybe store an retrieve are enough ?)

Creating a REPL:
This one seems very tricky as in a REPL memory and registers should be persistent for the session, so I mght have to implement it inside the VM as a method. Might need additional methods to check the status of the registers and the vRAM.
"""