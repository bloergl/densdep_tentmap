# densdep_tentmap
Simulation tools for density-dependent tent map dynamics

Simulation tools for iterating density dependent tent maps
==========================================================

This package provides two algorithms for numerically computing
the density dependent iterative tent map. These are provided as
functions in two Python modules:

sim_discrete.py:
----------------
Approximates density by a (typically large) set of samples.
This is a relatively trivial implementation whose precision,
however, is considerably limited by the number of samples.

sim_volume.py:
--------------
Represents density by a set of boxes whose total volume is 1.
The iteration algorithm 'stretches' and 'wraps' the boxes
according to the tent map. The precision of the algorithm
is limited only by the numerical precision with which the
volumes of the boxes and their locations are represented.
A second caveat is that each iteration of the map usually
adds a new box to the set, which makes progress slower with
increasing iteration depth.

The module also provides functions for plotting or further
processing of the results.

sim_tools.py:
-------------
Some tools which might be useful, e.g. for making a movie
from a set of image files.

example_basicusage.py:
----------------------
Demonstrates the basic usage of the two iteration methods, and
compares the results. In addition to showing the user how to
employ the modules, one can change the number of discrete values
used to represent the density and check how changing this
number impacts on the precision of the results (it turns out that
increasing n keeps the discrete sim close to the boxes sim for
a larger number of iterations, thus confirming that the boxes sim
is more precise/reliable)

example_returnmap.py:
---------------------
Computes the map a[n] -> a[n+1] for different values of the integral
parameter delta and for different computation methods (from 0.5 to 0.9).
When executing sim_tools.py directly, it uses the movie making function
to generate an animation of the corresponding results.



