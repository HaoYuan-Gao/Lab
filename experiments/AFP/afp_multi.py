import math

class AFP(object):
    def __init__(self, M: int, S: int, N: int, group_up: bool = False, mask_bits: int = 3):
        self.M = M # 4,5,6,7
        self.S = S # 1,2,3,4
        self.N = N # 1 2 3 4
        self.group_up = group_up
        self.mask_bits = mask_bits # default is 0
        self.MANTISSA_MIN = -128
        self.MANTISSA_MAX = 127

    def convert(self, value: float, return_float: bool = True):
        value = max(-1.0, min(1.0, value))
        
        if abs(value) < 1e-10:
            if return_float:
                return 0.0
            else:
                return [(0, 0)] if self.mask_bits > 0 else (0, 0)
        
        scale_factor = 2**(self.M+(self.S*(self.N-1)))-1
        quantized_int = round(value * scale_factor)

        if quantized_int == 2**(self.M+(self.S*(self.N-1))):
            quantized_int -= 1
        if quantized_int == -1 * 2**(self.M+(self.S*(self.N-1))):
            quantized_int += 1

        abs_quantized_int = abs(quantized_int)
        
        if abs_quantized_int == 0:
            significant_bit = 0
        else:
            significant_bit = math.floor(math.log2(abs_quantized_int))
        
        if self.group_up:
            exponent_step = significant_bit // self.S
            if exponent_step >= self.N:
                exponent_step = self.N - 1
        else:
            exponent_step = (significant_bit - self.M + self.S) // self.S
            if exponent_step < 0:
                exponent_step = 0

        # Branch based on mask_bits
        if self.mask_bits == 0:
            # Original exact logic - no group selection, direct calculation
            exponent = self.S * exponent_step
            mantissa = round(quantized_int / (2**exponent))
            mantissa = max(self.MANTISSA_MIN, min(self.MANTISSA_MAX, mantissa))
            
            quantized_value = mantissa * 2**(exponent) / scale_factor

            if return_float:
                return quantized_value
            else:
                return mantissa, exponent
        
        else:
            # Multi-group logic with masking and compensation
            pairs = []
            remaining_int = quantized_int
            
            for try_step in range(exponent_step, -1, -1):
                if remaining_int == 0:
                    break
                    
                exponent = self.S * try_step
                mantissa = round(remaining_int / (2**exponent))
                mantissa = max(self.MANTISSA_MIN, min(self.MANTISSA_MAX, mantissa))
                
                # Apply bit masking for groups > 0
                if try_step > 0:
                    mantissa = (mantissa >> self.mask_bits) << self.mask_bits
                
                if mantissa != 0:
                    pairs.append((mantissa, exponent))
                    represented_int = mantissa * (2**exponent)
                    remaining_int -= represented_int

            total_quantized_value = sum(mantissa * 2**exponent for mantissa, exponent in pairs) / scale_factor

            if return_float:
                return total_quantized_value
            else:
                return pairs
            
    def convert_cuda(self, x, return_extra: bool = False, max_pairs: int = 8):
        import cupy as cp
        import afp_ext

        # --- prepare input on GPU ---
        if not isinstance(x, cp.ndarray):
            x = cp.asarray(x, dtype=cp.float32)
        else:
            if x.dtype != cp.float32:
                x = x.astype(cp.float32, copy=False)

        if not x.flags.c_contiguous:
            x = cp.ascontiguousarray(x)

        y = cp.empty_like(x)

        M, S, N = int(self.M), int(self.S), int(self.N)
        group_up = 1 if self.group_up else 0
        mask_bits = int(self.mask_bits)
        mantissa_min = int(self.MANTISSA_MIN)
        mantissa_max = int(self.MANTISSA_MAX)
        n = int(x.size)

        if mask_bits == 0:
            m_single = cp.empty(n, dtype=cp.int32) if return_extra else None
            e_single = cp.empty(n, dtype=cp.int32) if return_extra else None

            afp_ext.multi_afp_convert(
                x.data.ptr, y.data.ptr,
                0 if m_single is None else m_single.data.ptr,
                0 if e_single is None else e_single.data.ptr,
                0, 0, 0,
                n,
                M, S, N, group_up, mask_bits, mantissa_min, mantissa_max,
                0  # stream_ptr: 0 = default stream
            )

            if return_extra:
                return y, (m_single, e_single)
            return y

        else:
            count = cp.empty(n, dtype=cp.int32) if return_extra else None
            m_multi = cp.empty((n, max_pairs), dtype=cp.int32) if return_extra else None
            e_multi = cp.empty((n, max_pairs), dtype=cp.int32) if return_extra else None

            afp_ext.multi_afp_convert(
                x.data.ptr, y.data.ptr,
                0, 0,
                0 if count is None else count.data.ptr,
                0 if m_multi is None else m_multi.data.ptr,
                0 if e_multi is None else e_multi.data.ptr,
                n,
                M, S, N, group_up, mask_bits, mantissa_min, mantissa_max,
                0
            )

            if return_extra:
                return y, (count, m_multi, e_multi)
            return y