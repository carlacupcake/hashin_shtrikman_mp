import ctypes

class CHashEntry(ctypes.Structure):
    pass

CHashEntry._fields_ = [
    ("key", ctypes.c_char_p),
    ("value", ctypes.c_char_p),
    ("next", ctypes.POINTER(CHashEntry))
]

class CHashTable(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("buckets", ctypes.POINTER(ctypes.POINTER(CHashEntry)))  # Array of pointers to CHashEntry
    ]


    

