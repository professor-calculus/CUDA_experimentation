# CUDA_experimentation
Playing around with CUDA in C/C++, learning the basics and trying to get better!

Compilation: nvcc <script>.cu -o <script>
Run with ./<script> or nvprof ./<script>

* sum_array.cu: This self-contained script sums the elements of an array recursively. Can handle arrays of any size up to the limits of the grid dimensions of one's card.

* max_array.cu: This script returns the maximum element of an array, iterating in much the same way as sum_array. Loops over blocks on GPU recursively, for arbitrary array size.
