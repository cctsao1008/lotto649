"""Shared six-line encoding and sector arithmetic."""



def gray3(index: int) -> int:
    if type(index) is not int or not 0 <= index < 8:
        raise ValueError("three-bit Gray sector must be 0..7")
    return index ^ (index >> 1)

def sector8(position: int, period: int) -> int:
    """Exact integer half-open binning: no floating-point threshold drift."""
    if type(position) is not int or type(period) is not int or period <= 0 or not 0 <= position < period:
        raise ValueError("invalid cyclical position/period")
    result = (8 * position) // period
    if not 0 <= result < 8:
        raise AssertionError("invalid sector")
    return result

def hexagram_from_sectors(solar_sector: int, lunar_sector: int) -> dict[str, str]:
    upper, lower = gray3(solar_sector), gray3(lunar_sector)
    code = (upper << 3) | lower
    return {
        "astronomical_hexagram_code": str(code),
        "astronomical_hexagram_binary_bottom_to_top": "".join(str((code >> i) & 1) for i in range(6)),
        "astronomical_lower_trigram_code": str(lower),
        "astronomical_upper_trigram_code": str(upper),
        "astronomical_yang_count": str(code.bit_count()),
    }


def outcome_code(nums: tuple[int, ...]) -> int:
    if (len(nums) != 6 or any(type(v) is not int or v < 1 or v > 49 for v in nums)
            or tuple(sorted(set(nums))) != nums):
        raise ValueError("six ascending unique integers in 1..49 required")
    return sum((num & 1) << pos for pos, num in enumerate(nums))
