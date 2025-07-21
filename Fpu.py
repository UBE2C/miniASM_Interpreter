from __future__ import annotations

import math
from typing import override

from Custom_errors import FpuError

class FPU_multiplier_divider:
    
    @override
    def __init__(self) -> None:
        self.external_methods: list[str] = ["multiply_floats", "divide_floats"]

    @override
    def __str__(self) -> str:
        return f"< The multiplier and divider subunit of the FPU with the external methods: {self.external_methods} >\n< Takes int and or float as an input and returns a float type. >"
    
    @override
    def __repr__(self) -> str:
        return self.__str__()
    

    #Internal methods
    #NOTE: Floating point multiplication and division shared internal methods start here
    def binary_to_float(self, fpn_bit_string: str, bit_len: int = 64) -> float:
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


    def full_adder(self, input_a: int = 0, input_b: int = 0, carry_in: int = 0) -> tuple[int, int]:
        """
        Hardware-level 1-bit full adder implementation.
        
        Performs binary addition of two single bits plus a carry input,
        producing a sum bit and carry output. This mimics actual CPU
        arithmetic logic unit (ALU) behavior for bit-accurate arithmetic.
        
        The implementation uses XOR for sum calculation and combines AND/OR
        logic for carry propagation, matching standard digital logic design.
        
        Args:
            input_a: First input bit (0 or 1, defaults to 0)
            input_b: Second input bit (0 or 1, defaults to 0) 
            carry_in: Carry input from previous bit position (0 or 1, defaults to 0)
            
        Returns:
            tuple[int, int]: (sum_bit, carry_out) where:
                - sum_bit: Result of XOR operation on all three inputs
                - carry_out: 1 if two or more inputs are 1, otherwise 0
                
        Note:
            All inputs are expected to be 0 or 1. The function implements
            the standard truth table for binary addition at the bit level.
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


    def float_rounder(self, exponent: str, mantissa: str, rounding_bits: str) -> tuple[str, str]:
        """
        IEEE 754 compliant rounding using Guard, Round, and Sticky bits.
        
        Implements "round to nearest, ties to even" (banker's rounding) with
        hardware-accurate bit manipulation. The function handles mantissa overflow
        by propagating carries through the exponent using a full adder chain.
        
        Rounding Algorithm:
        - If guard bit = 0: truncate (no rounding)
        - If guard bit = 1:
            - If round = 0 and sticky = 0: round to even (check LSB of mantissa)
            - If round = 1 or sticky = 1: always round up
        
        Implementation Details:
        - Reverses bit arrays for LSB-first processing during addition
        - Uses sticky bit calculation via OR reduction of all trailing bits
        - Employs hardware-level full adder for carry propagation
        - Handles mantissa overflow by incrementing exponent
        - Maintains bit-accurate arithmetic throughout the process
        
        Args:
            exponent: Binary string of the biased exponent (any length)
            mantissa: Binary string of the mantissa without implicit leading bit
            rounding_bits: Extra precision bits in format "GRS..." where:
                - G: Guard bit (position 0)
                - R: Round bit (position 1) 
                - S...: Sticky bits (positions 2+, OR'd together)
            
        Returns:
            tuple[str, str]: (rounded_exponent, rounded_mantissa)
                - If no rounding needed: returns original values
                - If rounding causes overflow: returns incremented exponent
                - If invalid state: returns ('', '') as error indicator
                
        Note:
            The function modifies bit arrays in-place using reversed order for
            efficient carry propagation, then reverses back for output format.
            Mantissa overflow automatically triggers exponent increment.
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
                        new_bit, carry_over = self.full_adder(input_a = bit, carry_in = 1)
                        rounded_mantissa.append(new_bit)
                        
                    else:
                        new_bit, carry_over = self.full_adder(input_a = bit, carry_in = carry_over)
                        rounded_mantissa.append(new_bit)

                if carry_over == 0:
                    rounded_mantissa.reverse()

                    rounded_mantissa_out = ''.join(str(bit) for bit in rounded_mantissa)
                    return exponent, rounded_mantissa_out

                else:
                    for bit_index, bit in enumerate(exponent_lst):
                        new_bit: int = 0

                        if bit_index == 0:
                            new_bit, carry_over = self.full_adder(input_a = bit, carry_in = 1)
                            rounded_exponent.append(new_bit)
                            
                        else:
                            new_bit, carry_over = self.full_adder(input_a = bit, carry_in = carry_over)
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
                    new_bit, carry_over = self.full_adder(input_a = bit, carry_in = 1)
                    rounded_mantissa.append(new_bit)
                        
                else:
                    new_bit, carry_over = self.full_adder(input_a = bit, carry_in = carry_over)
                    rounded_mantissa.append(new_bit)
            
            if carry_over == 0:
                rounded_mantissa.reverse()
                
                rounded_mantissa_out = ''.join(str(bit) for bit in rounded_mantissa)
                return exponent, rounded_mantissa_out

            else:
                for bit_index, bit in enumerate(exponent_lst):
                    new_bit: int = 0

                    if bit_index == 0:
                        new_bit, carry_over = self.full_adder(input_a = bit, carry_in = 1)
                        rounded_exponent.append(new_bit)
                                
                    else:
                        new_bit, carry_over = self.full_adder(input_a = bit, carry_in = carry_over)
                        rounded_exponent.append(new_bit)

                rounded_exponent.reverse()
                rounded_mantissa.reverse()

                rounded_exponent_out = ''.join(str(bit) for bit in rounded_exponent)
                rounded_mantissa_out = ''.join(str(bit) for bit in rounded_mantissa)

                return rounded_exponent_out, rounded_mantissa_out

        else:
            return '', ''         


    def float_to_binary(self, num: float | int, bit_len: int = 64) -> str:
        """
        Convert a floating point number to binary representation in IEEE 754 format.
        
        Implements complete IEEE 754 standard with hardware-accurate precision and
        rounding. Handles the full spectrum of IEEE 754 values including normal,
        subnormal, zero, infinity, and NaN cases with proper bit-level accuracy.
        
        IEEE 754 Support:
        - Normal numbers: Standard 1.mantissa × 2^(exponent-bias) format
        - Subnormal numbers: 0.mantissa × 2^(1-bias) format for values near zero
        - Special values: +/-0, +/-infinity, NaN with correct bit patterns
        - Precision modes: Single (32-bit) and double (64-bit) precision
        
        Algorithm Details:
        - Uses logarithmic calculation for normal number exponents
        - Implements subnormal scaling via 2^(bias-1) multiplication
        - Employs 5 extension bits for intermediate precision during calculation
        - Integrates IEEE 754 compliant rounding with guard/round/sticky bits
        - Validates output length and detects overflow/underflow conditions
        
        Special Value Handling:
        - Zero: Detected via direct comparison, returns appropriate zero pattern
        - NaN: Detected using self-inequality property (NaN != NaN)
        - Infinity: Detected via math.isinf(), returns infinity bit pattern
        - Sign: Handled uniformly across all value types
        
        Args:
            num: Float or int number to convert
            bit_len: Target bit length (32 for single, 64 for double precision)
            
        Returns:
            String representation of the binary number in IEEE 754 format
            Format: [sign][exponent][mantissa] with exact bit_len length
            
        Raises:
            ValueError: If bit_len is not 32 or 64
            BufferError: If overflow, underflow, or output length mismatch occurs
                - Overflow: When exponent exceeds maximum representable value
                - Underflow: When exponent is below minimum subnormal range
                - Length mismatch: When output doesn't match expected bit_len
                
        Implementation Notes:
        - Bit arrays are processed in LSB-first order for arithmetic operations
        - Mantissa extraction uses iterative fraction doubling algorithm
        - Subnormal detection uses threshold 2^(1-bias) for range boundaries
        - Extension bits provide extra precision before final rounding step
        - All intermediate calculations maintain bit-level accuracy
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

            biased_exp_bin, mantissa = self.float_rounder(exponent = biased_exp_bin, mantissa = mantissa, rounding_bits = rounding_bits)
            
        #Build the final output binary and check the output length
        output_bin_string = sign_bit + biased_exp_bin + mantissa

        if len(output_bin_string) != bit_len:
            raise BufferError(f"float_to_binary: the output bit len was expected to be {bit_len}, {len(output_bin_string)} was received")
    

        return output_bin_string


    def int_to_bits(self, input_int: int, bit_len: int = 64) -> list[int]:
        """
        Convert an integer to its binary representation as a list of bits.
        
        Creates a binary representation with the least significant bit (LSB) first,
        which facilitates easier bit-level arithmetic operations. The function applies
        a width mask to ensure the output is exactly the specified bit length.

        Args:
            input_int: The integer to convert to binary representation
            bit_len: The bit width to use (default: 64)
        
        Returns:
            A list of bits in LSB->MSB order (least significant bit first)
            
        Notes:
            - Applies a width mask to truncate values exceeding bit_len
            - LSB-first ordering simplifies carry propagation in arithmetic operations
            - Used extensively in floating-point exponent and mantissa processing
            
        Example:
            int_to_bits(5, 4) returns [1, 0, 1, 0] representing binary 0101
        """
            
        width_mask: int = (1 << bit_len) -1 #creates a mask of 64 1s to cut the target to 64 bits
        bit_seq: list[int] = []

        masked_input = input_int & width_mask

        for bit_index in range(bit_len):
            bit_seq.append((masked_input >> bit_index) & 1)

        return bit_seq


    def bit_to_int(self, input_bits: list[int], signed: bool) -> int:
        """
        Convert a list of bits to a signed or unsigned integer.
        
        Reconstructs an integer value from a bit list in LSB->MSB order.
        Handles both signed (two's complement) and unsigned interpretations.
        Critical for converting computed bit sequences back to numeric values
        during floating-point operations.

        Args:
            input_bits: The bit list to convert (LSB->MSB order)
            signed: If True, interprets MSB as sign bit using two's complement
        
        Returns:
            The integer value represented by the bit list
            
        Notes:
            - For signed=True: Uses two's complement representation
            - For signed=False: Standard unsigned binary interpretation
            - Essential for exponent bias calculations and overflow detection
            - Handles negative results correctly for subnormal exponent calculations
            
        Example:
            bit_to_int([1, 0, 1, 0], False) returns 5 (unsigned)
            bit_to_int([1, 1, 1, 1], True) returns -1 (signed two's complement)
        """
            
        bit_string: int = 0

        for i, bit in enumerate(input_bits):
            mask: int = bit << i #create a mask by pushing the given bit to the given position
            bit_string = bit_string | mask #OR the mash with the bit string

        #Check the signed mode
        if signed == True and input_bits[-1] == 1: #if in signed mode and the sign bit is 1 convert to 2's complement
            return bit_string - (1 << len(input_bits)) #bit string - the max position value (1024 for 10 bits, 8192 for 13 bits)
            
        else:
            return bit_string

    def fp_twos_complement(self, bit_seq: list[int]) -> list[int]:
        """
        Convert a bit sequence to its two's complement representation.
        
        Implements two's complement arithmetic by inverting all bits and adding 1.
        Essential for subtraction operations in floating-point arithmetic,
        particularly for bias subtraction in exponent calculations.
        Uses full_adder for proper carry propagation.

        Args:
            bit_seq: A bit sequence in LSB->MSB order to convert
        
        Returns:
            Two's complement representation as a bit list (LSB->MSB order)
            
        Notes:
            - Step 1: Invert all bits (one's complement)
            - Step 2: Add 1 using full_adder with carry propagation
            - Used primarily in sub_bias() for exponent bias subtraction
            - Maintains LSB->MSB ordering for arithmetic consistency
            
        Algorithm:
            1. Create one's complement by inverting each bit
            2. Add 1 to the inverted sequence using full_adder
            3. Propagate carry through all bit positions
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
                new_bit, carry_over = self.full_adder(input_a = bit, carry_in = 1)
                twoC_seq.append(new_bit)

            else:
                    
                new_bit, carry_over = self.full_adder(input_a = bit, carry_in = carry_over)
                twoC_seq.append(new_bit)

        return twoC_seq




    #NOTE: Floating point multiplication unique internal methods start here
    def add_biased_exponents(self, exponent_1: list[int], exponent_2: list[int], intermediate_len: int) -> list[int]:
        """
        Add two biased IEEE 754 exponents for floating-point multiplication.
        
        Performs binary addition of two exponent bit sequences with overflow detection.
        The exponents are added in their biased form (no bias removal required).
        This is the first step in the IEEE 754 multiplication formula: 
        new_exp = (exp1 + exp2) - bias.
        
        Args:
            exponent_1: First exponent as list of bits in MSB->LSB order
            exponent_2: Second exponent as list of bits in MSB->LSB order  
            intermediate_len: Target bit length for intermediate calculations (includes overflow protection)
            
        Returns:
            Sum of exponents as list of bits in LSB->MSB order, padded to intermediate_len
            
        Raises:
            FpuError: If arithmetic overflow is detected during addition
            
        Notes:
            - Input exponents are automatically zero-padded to intermediate_len
            - Reverses inputs to LSB->MSB for easier bit-by-bit addition
            - Uses full_adder for proper carry propagation
            - Overflow detection compares MSB input carry with final carry
            - Output remains in LSB->MSB order for subsequent bias subtraction
            
        Algorithm Flow:
            1. Copy and reverse input exponents (MSB->LSB to LSB->MSB)
            2. Zero-pad both exponents to intermediate_len
            3. Perform bit-by-bit addition with carry propagation
            4. Check for overflow by comparing MSB carry states
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

            new_bit, carry_over = self.full_adder(input_a = exp_1[bit_index], input_b = exp_2[bit_index], carry_in = carry_over)

            new_seq.append(new_bit)

        #Check for overflow
        if msb_in != carry_over:
            raise FpuError(message="add_biased_exponents: overflow detected at the end of exponent addition")

        return new_seq
        

    def sub_bias(self, exponent_seq: list[int], bias: int, intermediate_len: int, final_len: int, subnormal: bool) -> tuple[list[int], int]:
        """
        Subtract the IEEE 754 exponent bias from a biased exponent sum with subnormal handling.
        
        Completes the IEEE 754 exponent calculation: final_exp = (exp1 + exp2) - bias.
        Includes comprehensive subnormal number detection and handling:
        - Detects when results fall below the normal range
        - Calculates mantissa right-shift amounts for subnormal results
        - Handles overflow detection for infinity generation
        - Supports both normal and subnormal input number bias adjustments
        
        Args:
            exponent_seq: Biased exponent sum as list of bits in LSB->MSB order
            bias: IEEE 754 bias value (127 for 32-bit, 1023 for 64-bit)
            intermediate_len: Bit length of input exponent sequence
            final_len: Target bit length for final exponent output
            subnormal: If True, uses modified bias for subnormal input handling
            
        Returns:
            Tuple containing:
            - Unbiased exponent as list of bits in MSB->LSB order, trimmed to final_len
            - Mantissa shift amount (0 for normal, >0 for subnormal results)
            
        Raises:
            FpuError: If arithmetic overflow is detected during subtraction
            
        Notes:
            - For subnormal inputs: bias = 1 - (original_bias + leading_zeros)
            - For normal inputs: bias = original_bias
            - Subnormal detection: checks if result < (1 - bias)
            - Subnormal output: exponent = all zeros, shift = |result| + 1
            - Overflow detection: checks for bits beyond final_len
            
        Subnormal Handling Details:
            - Detects subnormal results by comparing unbiased exponent with minimum normal
            - Calculates mantissa right-shift using formula: |exp + bias| + 1
            - Returns all-zero exponent pattern for subnormal results
            - Maintains IEEE 754 compliance for gradual underflow
            
        Algorithm Flow:
            1. Create appropriate bias (normal or subnormal-adjusted)
            2. Convert bias to two's complement for subtraction
            3. Perform bit-by-bit subtraction with overflow detection
            4. Check for subnormal result and calculate shift if needed
            5. Check for overflow and return infinity pattern if needed
            6. Return final exponent and mantissa shift amount
        """
        

        if subnormal == True: #for subnormal numbers we use the Ex + Ey (done before) - (1-[127 + nlz]) formula to get a normalized subnormal exponent
            bias_seq: list[int] = self.int_to_bits(input_int = (1 - bias), bit_len = intermediate_len)
            bias_2c: list[int] = bias_seq #as the generated bias is negative, it is already in two's complement form after translation into a bit seq
            

        else:
            bias_seq: list[int] = self.int_to_bits(input_int = bias, bit_len = intermediate_len)
            bias_2c: list[int] = self.fp_twos_complement(bit_seq = bias_seq)

        carry_over: int = 0
        new_seq: list[int] = []
        msb_in: int = 0


        for bit_index in range(intermediate_len):
            new_bit: int = 0

            if bit_index == (intermediate_len - 1):
                msb_in = carry_over
                
            new_bit, carry_over = self.full_adder(input_a = exponent_seq[bit_index], input_b = bias_2c[bit_index], carry_in = carry_over)

            new_seq.append(new_bit)

        if msb_in != carry_over:
            raise FpuError(message="sub_bias: overflow detected at the end of bias subtraction")
        
        #Calculate the new unbiased exponent to check if the result is a subnormal number and define the mantissa shift subnormals
        biased_new_exponent: int = self.bit_to_int(input_bits = new_seq, signed = True) #this is the |ex+ey-bias| part of the shift calculation
        mantissa_shift: int = 0
        
        #detect a subnormal result by checking if the extended new biased exponent is less than the lowest normal exponent
        if biased_new_exponent - bias < (1-bias):
            output: list[int] = [0 for _ in range(final_len)] #Subnormal exponent pattern: all 0s
            output = output[::-1]
            
            
            #If the result is subnormal calculate the necessary mantissa shift  using the formula |ex+ey-bias| + 1
            mantissa_shift: int = abs(biased_new_exponent) + 1
            #mantissa_shift = mantissa_shift

            return output, mantissa_shift

        #detect overflow which should produce an Inf value
        if subnormal == False and new_seq[final_len : len(new_seq)].count(1) != 0: #there are 1s over the exponent bit limit
            output: list[int] = [1 for _ in range(final_len)] #Inf exponent pattern: all 1s
            output = output[::-1]
            

            return output, mantissa_shift
        
        else:
            output: list[int] = new_seq[0:final_len]
            output = output[::-1]


            return output, mantissa_shift

        
    def mant_multiplier(self, mantissa_1: list[int], mantissa_2: list[int], new_exponent: list[int], mantissa_length: int,
                        subn_multiplicand: bool, subn_multiplier: bool, nlz_multiplicand: int, nlz_multiplier: int,
                        subn_result: bool, subn_shift: int) -> tuple[list[int], list[int]]:
        """
        Multiply two IEEE 754 mantissas with comprehensive subnormal number support.
        
        Performs binary multiplication of two mantissas using shift-and-add algorithm.
        Includes sophisticated handling for subnormal numbers:
        - Automatic hidden bit restoration (1.xxx for normal, 0.xxx for subnormal)
        - Mantissa normalization through leading zero removal and padding
        - Subnormal result handling with appropriate right-shifting
        - Overflow detection and exponent adjustment for normalization
        
        The multiplication implements the mantissa portion of IEEE 754 multiplication:
        result_mantissa = mantissa_1 × mantissa_2 (with proper normalization)
        
        Args:
            mantissa_1: First mantissa (multiplicand) as list of fractional bits in MSB->LSB order
            mantissa_2: Second mantissa (multiplier) as list of fractional bits in MSB->LSB order
            new_exponent: Current exponent sum as list of bits in MSB->LSB order
            mantissa_length: Length of input mantissas (23 for 32-bit, 52 for 64-bit)
            subn_multiplicand: True if first operand is subnormal
            subn_multiplier: True if second operand is subnormal
            nlz_multiplicand: Number of leading zeros in first operand's mantissa
            nlz_multiplier: Number of leading zeros in second operand's mantissa
            subn_result: True if result should be treated as subnormal
            subn_shift: Right-shift amount for subnormal result normalization
            
        Returns:
            Tuple containing:
            - Updated exponent as list of bits in MSB->LSB order (incremented if overflow)
            - Final mantissa as list of fractional bits (hidden bit removed)
            
        Raises:
            AluError: If overflow is detected during partial product addition or exponent adjustment
            
        Subnormal Input Handling:
            - Normal numbers: Insert hidden bit 1, no shifting needed
            - Subnormal numbers: Insert hidden bit 0, then normalize by:
            1. Remove leading zeros (left shift simulation)
            2. Pad with zeros to maintain mantissa length
            - Mixed cases: Handle each operand according to its type
            
        Multiplication Algorithm:
            1. Restore appropriate hidden bits (1 for normal, 0 for subnormal)
            2. Normalize subnormal mantissas by removing leading zeros
            3. Perform shift-and-add multiplication
            4. Sum all partial products with overflow detection
            5. Apply subnormal result shifting if needed
            6. Check for mantissa overflow and adjust exponent
            
        Overflow and Normalization:
            - Checks MSB of result for overflow (mantissa ≥ 2.0)
            - If overflow: increment exponent, right-shift mantissa
            - If no overflow: remove overflow guard bit and hidden bit
            - Maintains IEEE 754 normalized form (1.xxx for normal results)
            
        Notes:
            - Result length is 2 × (mantissa_length + 1) bits during calculation
            - Supports variable precision through mantissa_length parameter
            - Handles all combinations of normal/subnormal operands
            - Implements proper IEEE 754 gradual underflow for subnormal results
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

            #pad them to normal length (2nd step of a left shift)
            [mant_1.append(0) for _ in range(nlz_multiplicand)]
            [mant_2.append(0) for _ in range(nlz_multiplier)]
        
        elif subn_multiplicand == True and subn_multiplier == False:
            mant_1.insert(0, 0) #subnormal, re-insert a 0
            mant_2.insert(0, 1) #normal, re-insert a 1

            #remove the leading 0s to normalize the mantissas (1st step of a left shift)
            mant_1 = mant_1[nlz_multiplicand : len(mant_1)]
            
            #pad them to normal length (2nd step of a left shift)
            [mant_1.append(0) for _ in range(nlz_multiplicand)]

        elif subn_multiplier == True and subn_multiplicand == False:
            mant_1.insert(0, 1) #normal, re-insert a 1
            mant_2.insert(0, 0) #subnormal, re-insert a 0

            #remove the leading 0s to normalize the mantissas (1st step of a left shift)
            mant_2 = mant_2[nlz_multiplier : len(mant_2)]

            #pad them to normal length (2nd step of a left shift)
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

                    new_bit, carry_over = self.full_adder(input_a = product_sum[bit_index], input_b = ip[bit_index], carry_in = carry_over)

                    new_mant_seq.append(new_bit)

                #Check for overflow between intermediate products
                if carry_over != 0:
                    raise FpuError(message="mant_multiplier: overflow detected during intermediate product addition")

                product_sum: list[int] = new_mant_seq #update the sum with the additional product
                

        #Reverse the new mantissa bit string back to MSB->LSB
        product_sum.reverse()
        
        if subn_result == True:
            #Check if the final output is a subnormal, if yes shift the mantissa accordingly
            for _ in range(subn_shift): #this will shift the bits string to the right, filling it with zeros
                product_sum.pop()
                product_sum.insert(0, 0)
            

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
                    new_bit, carry_over = self.full_adder(input_a = exp[bit_index], input_b = 0, carry_in = 1)

                else:
                    new_bit, carry_over = self.full_adder(input_a = exp[bit_index], input_b = 0, carry_in = carry_over)

                new_exp_seq.append(new_bit)

            #Check for overflow
            if msb_in != carry_over:
                raise FpuError(message="mant_multiplier: overflow detected during exponent adjustment")

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




    #NOTE: Floating point division unique internal methods start here
    def sub_biased_exponents(self, exponent_1: list[int], exponent_2: list[int], intermediate_len: int) -> list[int]:
        #Transfer ownership of the exponents
        exp_1: list[int] = exponent_1.copy()
        exp_2: list[int] = exponent_2.copy()

        #MSB -> LSB to LSB -> MSB conversion for easier calculations
        exp_1.reverse()
        exp_2.reverse()

        #Pad the exponents to the intermediate bit length 
        if len(exp_1) != intermediate_len or len(exp_2) != intermediate_len:
            exp_1.extend([0 for _ in range(intermediate_len - len(exp_1))])
            exp_2.extend([0 for _ in range(intermediate_len - len(exp_2))])

        #Transform the second exponent to two's complement
        exp_2 = self.fp_twos_complement(bit_seq = exp_2)

        #Declare variables
        new_seq: list[int] = []
        carry_over: int = 0
        msb_in: int = 0

        #Carry out the subtraction
        for bit_index in range(intermediate_len):
            new_bit: int = 0

            if bit_index == (intermediate_len - 1):
                msb_in: int = carry_over
            
            new_bit, carry_over = self.full_adder(input_a = exp_1[bit_index], input_b = exp_2[bit_index], carry_in = carry_over)
            
            new_seq.append(new_bit)

        #Check for overflow
        if msb_in != carry_over:
            raise FpuError(message="sub_biased_exponents: overflow detected at the end of exponent subtraction")

        return new_seq
        

    def add_bias(self, exponent_seq: list[int], bias: int, intermediate_len: int, final_len: int, subnormal: bool, subnormal_dividend: bool,
                 subnormal_divisor: bool, nlz_dividend: int, nlz_divisor: int) -> tuple[list[int], int]:
        #Borrow the new exponent seq
        exp: list[int] = exponent_seq.copy()

        #Set up exponent calculation based on the presence of subnormal operands the general formula is Ex-Ey+bias+Shift (Shift given by nlz)
        #NOTE: all the bias sequences for the subnormal numbers are adjusted bias sequences incorporating a +1 or -1 respectively
        if subnormal == False: 
            #no subnormals, formula: [Ex - Ey (done before)] + bias
            bias_seq: list[int] = self.int_to_bits(input_int = bias, bit_len = intermediate_len)
            #in the case of normal numbers there is no additional shift but I declare it to stop the language server from complaining
            shift: list[int] = self.int_to_bits(input_int = 0, bit_len = intermediate_len)  

        else: 
            if subnormal_dividend == True and subnormal_divisor == False:
                #the dividend is subnormal, formula:  - [Ex - Ey (done before)] + 1 + bias + nlz_dividend
                bias_seq: list[int] = self.int_to_bits(input_int = (1 + bias), bit_len = intermediate_len)
                bias_seq: list[int] = self.fp_twos_complement(bit_seq = bias_seq)
                shift: list[int] = self.int_to_bits(input_int = nlz_dividend, bit_len = intermediate_len)

            elif subnormal_dividend == False and subnormal_divisor == True:
                #the divisor is subnormal, formula: [Ex - Ey (done before)] − 1 + bias + nlz_divisor
                bias_seq: list[int] = self.int_to_bits(input_int = (-1 + bias), bit_len = intermediate_len)
                shift: list[int] = self.int_to_bits(input_int = nlz_divisor, bit_len = intermediate_len)

            elif subnormal_dividend == True and subnormal_divisor == True:
                #the divisor and dividend are subnormals, formula: bias + (nlz_dividend - nlz_divisor)
                bias_seq: list[int] = self.int_to_bits(input_int = bias, bit_len = intermediate_len)
                shift: list[int] = self.int_to_bits(input_int = (nlz_dividend - nlz_divisor), bit_len = intermediate_len)

            else:
                raise FpuError(message = f"add_bias: subnormal operand detected when none was supplied: subnormal {subnormal}")

        #Declare variables
        new_seq: list[int] = []
        carry_over: int = 0
        msb_in: int = 0

        for bit_index in range(intermediate_len):
            new_bit: int = 0

            if bit_index == (intermediate_len - 1):
                msb_in: int = carry_over

            new_bit, carry_over = self.full_adder(input_a = exp[bit_index], input_b = bias_seq[bit_index], carry_in = carry_over)

            new_seq.append(new_bit)

        #Check for overflow
        if msb_in != carry_over:
            raise FpuError(message="add_bias: overflow detected at the end of adding the bias")
        
        #shift if subnormal!
        if subnormal == True:
            shifted_exponent: list[int] = []
            carry_over: int = 0
            msb_in: int = 0

            for bit_index in range(intermediate_len):
                new_bit: int = 0

                if bit_index == (intermediate_len - 1):
                    msb_in: int = carry_over
                
                new_bit, carry_over = self.full_adder(input_a = new_seq[bit_index], input_b = shift[bit_index], carry_in = carry_over)

                shifted_exponent.append(new_bit)

            #Check for overflow
            if msb_in != carry_over:
                raise FpuError(message="add_biased: overflow detected at the end of exponent shift due to subnormal operands")

            #Swap the new exponent seq to the shifted exponent in the case of subnormal operands
            new_seq: list[int] = shifted_exponent

        #Calculate the new exponent to check if the result is a subnormal number and define the mantissa shift subnormals
        biased_new_exponent: int = self.bit_to_int(input_bits = new_seq, signed = True) #this is the |ex-ey+bias| part of the shift calculation
        mantissa_shift: int = 0
        
        #detect a subnormal result by checking if the extended new biased exponent is less than the lowest normal exponent
        if biased_new_exponent - bias < (1-bias):
            output: list[int] = [0 for _ in range(final_len)] #Subnormal exponent pattern: all 0s
            output: list[int] = output[::-1]
            
            #If the result is subnormal calculate the necessary mantissa shift  using the formula |ex-ey+bias| + 1
            mantissa_shift: int = abs(biased_new_exponent) + 1

            return output, mantissa_shift

        #detect overflow which should produce an Inf value
        if subnormal == False and new_seq[final_len : len(new_seq)].count(1) != 0: #there are 1s over the exponent bit limit
            output: list[int] = [1 for _ in range(final_len)] #Inf exponent pattern: all 1s
            output: list[int] = output[::-1]
            

            return output, mantissa_shift
        
        else:
            output: list[int] = new_seq[0:final_len]
            output: list[int] = output[::-1]


            return output, mantissa_shift


    def mantissa_divider(self, mantissa_1: list[int], mantissa_2: list[int], new_exponent: list[int], mantissa_length: int,
                         subn_dividend: bool, subn_divisor: bool, nlz_dividend: int, nlz_divisor: int,
                         subn_result: bool, subn_shift: int) -> tuple[list[int], list[int]]:


    #External method
    def multiply_floats(self, multiplicand: float | int, multiplier: float | int, precision: int = 64) -> float:
        """
        Multiply two floating-point numbers using IEEE 754 compliant binary arithmetic with comprehensive subnormal support.
        
        Implements complete IEEE 754 floating-point multiplication including:
        - Full subnormal number detection, normalization, and result handling
        - Exponent addition with precision-specific bias handling
        - Mantissa multiplication with hidden bit management and overflow detection
        - Gradual underflow implementation for subnormal results
        - Proper rounding with guard/round/sticky bits
        - Special value handling (zero, infinity, NaN)
        - Sign bit calculation (XOR of input signs)
        - Support for both single (32-bit) and double (64-bit) precision
        
        The multiplication follows the IEEE 754 formula:
        Result = (±1) × (normalized_mant1 × normalized_mant2) × 2^(exp1 + exp2 - bias)
        
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
            
        Special Cases Handled:
            - Zero multiplication: Returns correctly signed zero
            - Infinity multiplication: Returns correctly signed infinity
            - Zero × Infinity: Returns NaN (Not a Number)
            - Subnormal inputs: Proper normalization and bias adjustment
            - Subnormal results: Gradual underflow with mantissa right-shifting
            - Exponent overflow: Returns correctly signed infinity
            
        Subnormal Number Support:
            - Input Detection: Identifies subnormal operands (exp=0, mantissa≠0)
            - Normalization: Counts leading zeros and adjusts bias accordingly
            - Result Handling: Implements gradual underflow for subnormal products
            - Mantissa Shifting: Applies appropriate right-shifts for subnormal results
            - Precision Preservation: Maintains maximum possible precision during transitions
            
        Precision Specifications:
            32-bit: 8-bit exponent, 23-bit mantissa, bias=127
            64-bit: 11-bit exponent, 52-bit mantissa, bias=1023
            
        Algorithm Flow:
            1. Input validation and type checking
            2. Convert inputs to binary IEEE 754 representation
            3. Handle special cases (zero, infinity, NaN)
            4. Detect and process subnormal inputs
            5. Calculate new exponent: (exp1 + exp2) - bias
            6. Perform mantissa multiplication with normalization
            7. Apply subnormal result handling if needed
            8. Round result using extended precision
            9. Assemble final IEEE 754 bit pattern
            10. Convert back to native float format
            
        Notes:
            - Uses extended precision for intermediate calculations to prevent precision loss
            - Implements proper IEEE 754 rounding for final result
            - Maintains bit-level accuracy throughout computation
            - Handles all edge cases according to IEEE 754 standard
            - Subnormal handling ensures gradual underflow behavior
            - Comprehensive overflow detection prevents incorrect results
        """

        #Argument type checks
        if not isinstance(multiplicand, (float, int)) or not isinstance(multiplier, (float, int)):
            raise TypeError(f"float_multiplier: the arguments num_1 and num_2 must be of type float or int, {isinstance(num_1, (float, int))} and {isinstance(num_2, (float, int))} were provided.")

        if not isinstance(precision, (int)):
            raise TypeError(f"float_multiplier: the argument precision must be of type int, {isinstance(precision, (int))} was provided.")


        #Convert the inputs to bit strings
        num_1_bits: str = self.float_to_binary(num = multiplicand, bit_len = precision)
        num_2_bits: str = self.float_to_binary(num = multiplier, bit_len = precision)


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

            return self.binary_to_float(fpn_bit_string = float_out_bit_string, bit_len = precision)
        

        #Normal zero multiplication check and exit upon 0 multiplier or multiplicand
        if n1_bit_lst.count(1) == 0 or n2_bit_lst.count(1) == 0:
            final_exponent: str = "0" * exp_len #exponent must be all 0s
            
            final_mantissa: str = "0" * mant_len #mantissa must be all 0s

            new_sign_bit = n1_bit_lst[0] ^ n2_bit_lst[0]
            final_sign_bit: str = str(new_sign_bit)

            float_out_bit_string: str = final_sign_bit + final_exponent + final_mantissa

            return self.binary_to_float(fpn_bit_string = float_out_bit_string, bit_len = precision)

        #Input infinity check and exit upon exponent overflow
        if n1_bit_lst[1 : exp_len + 1].count(0) == 0 or n2_bit_lst[1 : exp_len + 1].count(0) == 0: #only 1s, no 0s
            final_exponent: str = ""
            for bit in range(exp_len):
                final_exponent += str(bit)

            final_mantissa: str = "0" * mant_len #mantissa must be all 0s

            new_sign_bit = n1_bit_lst[0] ^ n2_bit_lst[0]
            final_sign_bit: str = str(new_sign_bit)

            float_out_bit_string: str = final_sign_bit + final_exponent + final_mantissa
            

            return self.binary_to_float(fpn_bit_string = float_out_bit_string, bit_len = precision)
        

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

        if num_1_exp.count(1) == 0 and (multiplicand_mantissa.count(1) != 0):
            subnormal_multiplicand: bool = True
            subnormal_num: bool = True
            for bit in multiplicand_mantissa:
                if bit == 0:
                    nlz_multiplicand += 1
                else:
                    break

            nlz_multiplicand += 1 #add the hidden leading 0

        if num_2_exp.count(1) == 0 and (multiplier_mantissa.count(1) != 0):
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
        exponent_sum: list[int] = self.add_biased_exponents(exponent_1 = num_1_exp, exponent_2 = num_2_exp, intermediate_len = intermediate_buffer_len)
        #NOTE: for subnormal numbers the exponent bias will be 1 - (bias (-126 or -1022) + number of leading zeros) which should not effect normal number calculations as /
        #  1-bias is on a different if branch while nlz will be 0 for a normal number
        if subnormal_multiplicand == True and subnormal_multiplier == True:
            new_exponent, mantissa_shift = self.sub_bias(exponent_seq = exponent_sum, bias = (exp_bias + nlz_multiplicand + nlz_multiplier), intermediate_len = intermediate_buffer_len, final_len = exp_len, subnormal = subnormal_num)
        
        elif subnormal_multiplicand == True and subnormal_multiplier == False:
            new_exponent, mantissa_shift = self.sub_bias(exponent_seq = exponent_sum, bias = (exp_bias + nlz_multiplicand), intermediate_len = intermediate_buffer_len, final_len = exp_len, subnormal = subnormal_num)
            
        elif subnormal_multiplicand == False and subnormal_multiplier == True:
            new_exponent, mantissa_shift = self.sub_bias(exponent_seq = exponent_sum, bias = (exp_bias + nlz_multiplier), intermediate_len = intermediate_buffer_len, final_len = exp_len, subnormal = subnormal_num)

        else:
            new_exponent, mantissa_shift = self.sub_bias(exponent_seq = exponent_sum, bias = exp_bias, intermediate_len = intermediate_buffer_len, final_len = exp_len, subnormal = subnormal_num)

        
        #Check if we got a subnormal product during exponent calculation
        subnormal_product: bool = False
        if new_exponent.count(1) == 0:
            subnormal_product = True


        #Calculate the new, full length mantissa product and the potential new exponent
        new_exponent, mantissa_product = self.mant_multiplier(mantissa_1 = multiplicand_mantissa, mantissa_2 = multiplier_mantissa, new_exponent = new_exponent, mantissa_length = mant_len,
                                                        subn_multiplicand = subnormal_multiplicand, subn_multiplier = subnormal_multiplier, nlz_multiplicand = nlz_multiplicand, 
                                                        nlz_multiplier = nlz_multiplier, subn_result = subnormal_product, subn_shift = mantissa_shift)

        
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
        

        #Round the exponent and mantissa
        final_exponent, final_mantissa = self.float_rounder(exponent = exponent_string, mantissa = mantissa_string, rounding_bits = rounding_bits)

        
        #Infinity check for the final exponent value and exit upon exponent overflow
        if final_exponent.rfind("0") == -1: #only 1s, no 0s
            final_exponent: str = ""
            for bit in new_exponent:
                final_exponent += str(bit)

            final_mantissa: str = "0" * mant_len #mantissa must be all 0s

            new_sign_bit = n1_bit_lst[0] ^ n2_bit_lst[0]
            final_sign_bit: str = str(new_sign_bit)

            float_out_bit_string: str = final_sign_bit + final_exponent + final_mantissa
            

            return self.binary_to_float(fpn_bit_string = float_out_bit_string, bit_len = precision)
        

        #Decide the sign bit using the xor operation
        new_sign_bit = n1_bit_lst[0] ^ n2_bit_lst[0]
        final_sign_bit: str = str(new_sign_bit)


        #Assemble the new floating point number as a bit string
        float_out_bit_string: str = final_sign_bit + final_exponent + final_mantissa


        #Convert the new floating point bit string into a floating point number
        float_out: float = self.binary_to_float(fpn_bit_string = float_out_bit_string, bit_len = precision)

        return float_out
    

    def divide_floats(self, dividend: float | int, deviser: float | int, precision: int = 64) -> float:
        #Argument type checks
        if not isinstance(multiplicand, (float, int)) or not isinstance(multiplier, (float, int)):
            raise TypeError(f"float_multiplier: the arguments num_1 and num_2 must be of type float or int, {isinstance(num_1, (float, int))} and {isinstance(num_2, (float, int))} were provided.")

        if not isinstance(precision, (int)):
            raise TypeError(f"float_multiplier: the argument precision must be of type int, {isinstance(precision, (int))} was provided.")


        #Convert the inputs to bit strings
        num_1_bits: str = self.float_to_binary(num = multiplicand, bit_len = precision)
        num_2_bits: str = self.float_to_binary(num = multiplier, bit_len = precision)


        #Convert the bit strings to lists
        n1_bit_lst: list[int] = [int(i) for i in num_1_bits]
        n2_bit_lst: list[int] = [int(i) for i in num_2_bits]


        #Check if any operand is a subnormal 


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
            raise ValueError(f"divide_floats: 32 or 64 was expected for precision, {precision} was provided.")
        
        #Separate the exponent and mantissa bits
        num_1_exp: list[int] = n1_bit_lst[1 : exp_len + 1]
        num_2_exp: list[int] = n2_bit_lst[1 : exp_len + 1]

        dividend_mantissa: list[int] = n1_bit_lst[exp_len + 1 : (exp_len + 1) + mant_len] #bit 9 -> bit 32 in a 32 bit float (bit 32 is exclusive)
        divisor_mantissa: list[int] = n2_bit_lst[exp_len + 1 : (exp_len + 1) + mant_len]


        #Check for subnormal numbers and normalize them if present
        subnormal_num: bool = False
        subnormal_dividend: bool = False
        subnormal_divisor: bool = False
        
        nlz_dividend: int = 0 #number of leading 0s which is given by the mantissa
        nlz_divisor: int = 0 #number of leading 0s which is given by the mantissa

        if num_1_exp.count(1) == 0 and (dividend_mantissa.count(1) != 0):
            subnormal_dividend: bool = True
            subnormal_num: bool = True
            for bit in dividend_mantissa:
                if bit == 0:
                    nlz_dividend += 1
                else:
                    break

            nlz_dividend += 1 #add the hidden leading 0

        if num_2_exp.count(1) == 0 and (divisor_mantissa.count(1) != 0):
            subnormal_divisor: bool = True
            subnormal_num: bool = True
            for bit in divisor_mantissa:
                if bit == 0:
                    nlz_divisor += 1
                else:
                    break
            
            nlz_divisor += 1 #add the hidden leading 0


class FPU:
    def __init__(self, register_supervisor: "RegisterSupervisor | None" = None) -> None:  
        self.input_table: dict[str, bytearray] = {}


        self.last_op: str = ""
        self.last_output: bytearray = bytearray()
        self.last_output_type: str = ""
            
        self.register_supervisor: "RegisterSupervisor | None" = register_supervisor
        self.multiplier_divider: FPU_multiplier_divider = FPU_multiplier_divider()

            
        self.numeric_operations: set[str] = {"fadd", "fsub", "fmul", "fdiv", "finc", "fdec"}
            

    @override
    def __str__(self) -> str:
        return f"< ALU: last operation {self.last_op} with an output {self.last_output} of type {self.last_output_type} >"
        
    @override
    def __repr__(self) -> str:
        return self.__str__()



from Registers import RegisterSupervisor