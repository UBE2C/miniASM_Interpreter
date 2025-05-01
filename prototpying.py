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