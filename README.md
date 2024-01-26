# hashin_shtrikman_mp

mpi4py support is added to append final_dict

To take benefit of mpi parallelization, one can run the script as follows:

Install mpi4py via pip:
pip install mpi4py

In case installation via pip fails, you can use brew + pip instead:
brew install mpi4py
pip install mpi4py

Run the script as:
mpiexec -n 4 python orp_test.py