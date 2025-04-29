# mini_assembler_vm/__init__.py

# --- expose your core classes & functions at the package level ---
from .Token       import Token
from .Instruction import Instruction
from .lexer        import lexer
from .parser       import parser
from .Registers    import Registers
from .VirtualMachine           import VirtualMachine

# -- optional: exactly what 'from mini_assembler_vm import *' will grab --
__all__ = [
    "Token",
    "Instruction",
    "lexer", "parser",
    "Registers",
    "VirtualMachine",
]