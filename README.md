# The miniASM-VM

### The main idea
This is a fun attempt to build a fully functioning virtual machine and a minimal 64-bit CPU—with all its subcomponents and a dedicated RAM module—complete with its own type system, by modeling each component in Python.

For some time now, I've been fascinated by how hardware—especially the CPU—carries out various operations. Therefore, I decided I want to learn this stuff, and what better way is there to do that than to build one yourself?
If you are interested in building a "not too serious" VM and having some fun while doing it, you are very welcome to join in.
(Plus you can be a noob with me for doing this in Python and not C or C++ :))

### Progress so far:
- Built a functioning set of registers with a dedicated manager and associated methods.
- Built a functioning RAM module with a custom pointer and associated methods.
- Partially built an ALU — only addition and subtraction are implemented so far.
- Partially built an FPU — only floating-point multiplication and division are implemented so far.
- Partially integrated various components into a somewhat functioning VM for testing.

### Main focus for now:
- Finishing the FPU
- Finishing the ALU

### To do (rough goals):
- Finish the ALU by implementing multiplication, division, and logical operations.
- Finish the FPU by implementing floating-point addition and subtraction.
- Build and implement a program counter.
- Build and implement a decoder.
- Design the instruction lengths and the associated opcodes.
- Re-design the assembly intended to be used on the VM:
    + Re-design the instruction set.
    + Implement separate sections such as `.data` and `.code`.
- Re-design and re-write the Lexer, Parser, and Semantic Analyzer.
- Integrate the various components into a functional VM module.
- Figure out and implement testing for each component.
- Design and implement an "Interpret" and a "REPL" mode.

### Courses on these topcis I follow
Ross McGowan <br />
    - Design a Floating Point Unit series (https://www.udemy.com/course/design-a-floating-point-unit-1-numbers) <br />
    - Design a CPU series (https://www.udemy.com/course/design-a-cpu)

### Inspirations
I would like to say a huge thank you to the following games/creators (in no particular order) for sparking my interest in low-level concepts and inspiring me to explore how a CPU works:

**Games:**
- Turing Complete — an amazing game, check it out! :)
- Shenzhen I/O — another banger of a game, give it a spin!

**CodeWars Katas:** <br />
ShinuToki - Assembler interpreter series (https://www.codewars.com/kata/simple-assembler-interpreter/) <br />
Donaldsebleung - RoboScript series (https://www.codewars.com/kata/58708934a44cfccca60000c4) <br />

**YouTube creators:** <br />
<br />
CoreDumped (https://www.youtube.com/@CoreDumpped) <br />
ThePrimagen (https://www.youtube.com/@ThePrimeTimeagen) <br />
Casey Muratory (https://www.youtube.com/@caseymuratori) <br />
Low Level (https://www.youtube.com/@LowLevelTV) 
