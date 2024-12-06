# High Performance Linear Algebra
Luke Venkataramanan, Alex Lozano

## Build Instructions
In order to simplify compilation, building, and ensuring the program runs on the correct number of threads, we decided to use a build script called build.sh.

In order to run the program, create a new terminal. Navigate to the project root directory, and enter the following:

*chmod +x build.sh*

*./build.sh*

## Description
Linear algebra plays a foundational role in computational engineering and science, with applications ranging from solid state physics to machine learning. While these linear algebra operations are relatively straight-forward to naively implement, these approaches usually lack efficiency. In computational engineering, where performance is critical and problems are usually large scale, it is important to rely on methods that maximize performance.

This paper focuses on increasing the performance of matrix multiplication, a fundamental operation in linear algebra. As described in the next section, the naive implementation of a matrix-matrix product requires a triple nested loop, which has a time complexity of \(O(N^3)\), assuming the matrices have dimensions N-by-N. While this is acceptable for smaller matrices, this is impractical for matrices whose dimensions reach millions or billions. At such large scales, optimization strategies are essential. 

We take advantage of how data is stored in the computer memory in order to increase the speed of matrix multiplication, especially when the sizes of these matrices are very large. Specifically we found that using a recursive block multiplication technique, which is an example of cache-oblivious programming, significantly improves the performance of computing a matrix-matrix product.
