hipBone
=======

This is the `hipBone` repository.  `hipBone` is a GPU port of the original proxy
application called [`Nekbone`](https://github.com/Nek5000/Nekbone).

It solves a screened Poisson equation in a box using a conjugate gradient
method.

How to compile `hipBone`
------------------------

There are a couple of prerequisites for building `hipBone`;

- An MPI stack.  Any will work;
- OpenBlas.

Installing `MPI` and `OpenBlas` can be done using whatever package manager your
operating system provides.

To build and run `hipBone`, there is an included `run.sh` script which will
build the third party `OCCA`, then build `hipBone`, and run
several problem sizes and output figures of merit.

To build `hipBone` manually:

    $ git clone --recursive <hipBone repo>
    $ cd /path/to/hipBone
    $ export OPENBLAS_DIR=/path/to/openblas
    $ make -j `nproc`

How to run `hipBone`
--------------------

Here is an example CORAL-2 problem size that you can run on one GPU:

    $ mpirun -np 1 ./hipBone -m HIP -nx 24 -ny 24 -nz 24 -p 14

Here is the meaning of each of the command line options

- `nx`: the number of spectral elements in the x-direction per MPI rank
- `ny`: the number of spectral elements in the y-direction per MPI rank
- `nz`: the number of spectral elements in the z-direction per MPI rank
- `p`: the order of the polynomial used to approximate the solution
- `m`: the mode to run OCCA in, `HIP` is for AMD GPUs but `CUDA` and `Serial`
are also supported

Running on multiple GPUs can by done by passing a larger argument to `np` and
specifying the number of MPI ranks in each coordinate direction:

    $ mpirun -np 2 ./hipBone -m HIP -nx 24 -ny 24 -nz 24 -px 2 -py 1 -pz 1 -p 14

You must specify either:

1.  All of `px`, `py`, `pz`, or
2.  None of `px`, `py`, or `pz`.

If all of `px`, `py` and `pz` are specified then the product `px*py*pz` must
equal the argument passed to `np`.   If none of `px`, `py` or `pz` are
specified then the `np` must be a cube and `hipBone` will use an equal number
of MPI ranks in each coordinate direction.

Verifying correctness
---------------------

To verify that the computation is correct, add the `-v` option to the command
line.  Example output towards the end of the run may look like this:

    CG: it 96, r norm 1.328996666475e-19, alpha = 5.291357e-01
    CG: it 97, r norm 2.552900554560e-19, alpha = 1.990951e+00
    CG: it 98, r norm 3.836827649728e-19, alpha = 3.269689e+00
    CG: it 99, r norm 2.629545869383e-19, alpha = 1.509263e+00
    CG: it 100, r norm 2.045530932453e-19, alpha = 8.445030e-01
    hipBone: 3, 2744, 0.0249, 100, 9.08e-06,  3.7,  2.3, 1.10e+07; N, DOFs, elapsed, iterations, time per DOF, avg BW (GB/s), avg GFLOPs, DOFs*iterations/ranks*time
    hipBone: NekBone FOM =  2.6 GFLOPs.

The printed value of `r norm` at the end of 100 CG iterations should be small.

As per the [Nekbone CORAL-2 Benchmark summary](https://asc.llnl.gov/sites/asc/files/2020-06/Nekbone_Summary_v2.3.4.1.pdf):

> Benchmark results are considered correct if the reported r norm is small,
  generally less than 1e-8, after 100 conjugate gradient iterations.

How to clean build objects
--------------------------

To clean the `hipBone` build objects:

    $ cd /path/to/hipBone/repo
    $ make realclean

To clean JIT kernel objects:

    $ cd /path/to/hipBone/repo
    $ rm -r .occa

Please invoke `make help` for more supported options.
