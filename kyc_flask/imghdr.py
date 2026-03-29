"""
imghdr.py — compatibility shim for Python 3.13+
imghdr was removed from the standard library in Python 3.13.
This stub restores just enough for openbharatocr to import successfully.
"""

import struct


def what(file, h=None):
    """Identify the image type from file path or header bytes."""
    if h is None:
        if isinstance(file, str):
            with open(file, "rb") as f:
                h = f.read(32)
        else:
            location = file.tell()
            h = file.read(32)
            file.seek(location)

    # PNG
    if h[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    # JPEG
    if h[:2] == b"\xff\xd8":
        return "jpeg"
    # GIF
    if h[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    # BMP
    if h[:2] == b"BM":
        return "bmp"
    # TIFF
    if h[:2] in (b"MM", b"II"):
        return "tiff"
    # WebP
    if h[:4] == b"RIFF" and h[8:12] == b"WEBP":
        return "webp"
    # PPM / PGM / PBM
    if h[:2] in (b"P1", b"P2", b"P3", b"P4", b"P5", b"P6"):
        return "ppm"

    return None
