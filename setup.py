import sys
from cx_Freeze import setup, Executable

base = None
if sys.platform == "win32":
    base = "Win32GUI"

options = {"build_exe": {"includes": "atexit"}}

executables = [Executable("Penob.py", base=base)]

setup(
    name="Aplikasi B1",
    version="0.1",
    description="Aplikasi Pendeteksi Objek Menggunakan Otsu Thresholding",
    options=options,
    executables=executables,
)