# None-Negative Kernel-based Orthogonal Matching Pursuit (NN-KOMP)

This code solves the following problem with Cholesky decomposition: 

`X=argmin_X ||phi(y)-phi(Y) A X||_2^2`

`subject to x >= 0 , ||x||_0 < T0`
   

## Using KNNLS
- Try nnkomp_demo.m for a simple example for the proper usage of NN-KOMP.
