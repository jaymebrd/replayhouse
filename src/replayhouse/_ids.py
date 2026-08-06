import os
import time


def uuid7() -> str:
    """RFC 9562 UUIDv7: 48-bit unix ms timestamp + random, time-ordered."""
    b = bytearray(int(time.time_ns() // 1_000_000).to_bytes(6, "big") + os.urandom(10))
    b[6] = (b[6] & 0x0F) | 0x70
    b[8] = (b[8] & 0x3F) | 0x80
    h = bytes(b).hex()
    return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"
