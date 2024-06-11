## How to use the cython code
1. This code was developed with python 3.9 and has not been tested with later python versions.
2. From the `core` directory, run `python setup.py build_ext`.
4. For each file `./cbuilds/*.pyx`, this will create `*.c`, `*.cpp`, and `*.cpython-39-darwin.so`.

## Implementation notes
Cython classes and Pydantic classes cannot be one-in-the-same. That is why we have only cythonized some methods from the Pydantic model. This way, the performance-critical methods can still be cythonized while keeping the Pydantic model intact.
