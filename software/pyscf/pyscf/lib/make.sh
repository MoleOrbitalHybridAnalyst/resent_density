rm -rf *so deps/ build/
mkdir build
cd build
module load gcc/9.2.0 mkl-2017.0.098
export LD_PRELOAD=$MKLROOT/lib/intel64/libmkl_def.so:$MKLROOT/lib/intel64/libmkl_sequential.so:$MKLROOT/lib/intel64/libmkl_core.so
cmake -DCMAKE_C_COMPILER=gcc ..
make -j 20
