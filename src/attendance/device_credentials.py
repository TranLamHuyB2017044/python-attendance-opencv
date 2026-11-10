"""Local camera provisioning stored outside the Windows app and updater."""
import ctypes
import json
import os
import tempfile
from ctypes import wintypes
from pathlib import Path


class DataBlob(ctypes.Structure):
    _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_ubyte))]


def _dpapi(data, protect):
    if os.name != "nt":
        raise OSError("Camera credential storage requires Windows DPAPI")
    buffer = ctypes.create_string_buffer(data)
    source = DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    output = DataBlob()
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    # User-scoped and noninteractive. Provision using the camera process's account.
    transform = crypt32.CryptProtectData if protect else crypt32.CryptUnprotectData
    transform.argtypes = [
        ctypes.POINTER(DataBlob), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(DataBlob),
    ]
    transform.restype = wintypes.BOOL
    if not transform(ctypes.byref(source), None, None, None, None, 0x01, ctypes.byref(output)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return ctypes.string_at(output.data, output.size)
    finally:
        kernel32.LocalFree(output.data)


def protect(data):
    return _dpapi(data, protect=True)


def unprotect(data):
    return _dpapi(data, protect=False)


def camera_data_dir():
    base = os.getenv("LOCALAPPDATA")
    if not base:
        base = str(Path.home() / "AppData" / "Local")
    return Path(base) / "Bittech" / "Attendance"


def read_protected(name):
    path = camera_data_dir() / name
    return json.loads(unprotect(path.read_bytes())) if path.exists() else None


def write_protected(name, value):
    folder = camera_data_dir()
    folder.mkdir(parents=True, exist_ok=True)
    protected = protect(json.dumps(value).encode("utf-8"))
    handle, temporary = tempfile.mkstemp(dir=folder)
    try:
        with os.fdopen(handle, "wb") as file:
            file.write(protected)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, folder / name)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_device():
    return read_protected("device.dat")


def store_device(key, identity):
    write_protected("device.dat", {"api_key": key, **identity})
