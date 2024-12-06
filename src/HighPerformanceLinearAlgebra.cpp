// Names: Alexander Lozano and Luke Venkataramanan
// UT EID: AJL4846 and LV8828
// TACC: alozano0304 and luke_venk

#include <iostream>
#include <tuple>
#include <vector>
#include <span>
#include <random>
#include <algorithm>
#include <iterator>
#include <chrono>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "mdspan/mdspan.hpp"
namespace md = Kokkos;
using namespace std;

/* Use the DEBUG macro to use .at() for bounds-checking
 If this is commented out, we'll just use normal indexing
*/ 
// #define DEBUG

// Define the extents type by using int for index type, with a run-time number for M rows and N cols,
// which are dynamically defined at run-time
using extents_T = md::extents<int, std::dynamic_extent, std::dynamic_extent>;

class Matrix {
private:
    int M;  // The number of rows
    int N;  // The number of columns
    int LDA;  // The leading dimension of A
    md::mdspan<double, extents_T, md::layout_left> data;  // The data within the matrix (layout_left corresponds to column-major ordering)
    int topLeftIndex;  // The index corresponding to the matrix's global position within the larger memory layout

public:
    /**
     * @brief: Constructor for the Matrix class
     * @param m: The number of rows, M
     * @param n: The number of columns, N
     * @param lda: The leading dimension of A, LDA
     * @param dataIn: The pointer to the C style double array storing the data of the matrix
     */
    Matrix(int m, int n, int lda, double* dataPtr, int tlIndex = 0) : M(m), N(n), LDA(lda), 
						   data(dataPtr, extents_T(m, n)), topLeftIndex(tlIndex) {};

    /**
     * @brief: Get function: given row i and column j, return the data at that location
     * @param i: Row number
     * @param j: Column number
     * 
     * @return: The value of the element at location (i, j)
     */
    double& at(int i, int j) {
        // If in debug mode, do a bounds check
        #ifdef DEBUG
            if (i >= M || j >= N) {
                throw out_of_range("Error: Index out of bounds");
            }
        #endif
        // Use the previously discussed formula to calculate the linearized index of the element
        // in the context of the actual memory layout of larger matrix A, using the LDA
	    int linearIndex = topLeftIndex + (i + j*LDA);
        // Solve backwards for the rowIndex and colIndex by using div and mod M
        // in the context of the smaller matrix B (explanation in write up)
        int rowIndex = linearIndex % M;
        int colIndex = linearIndex / M;
        // data_handle() returns pointer to raw data
        // mapping() uses internal mapping function to convert 2D indexing to linear index based on extents_T
        return *(data.data_handle() + data.mapping()(rowIndex, colIndex));
    }

    /**
     * @brief: Set function: given row i, column j, set that element to the given value
     * @param i: Row number
     * @param j: Column number
     * @param v: Value that you want to set the matrix's (i, j) element to
     * 
     */
    void set(int i, int j, double v) {
        // Uses the same indexing scheme as the get function
        int linearIndex = topLeftIndex + (i + j*LDA); 
        int rowIndex = linearIndex % M;
        int colIndex = linearIndex / M;
        *(data.data_handle() + data.mapping()(rowIndex, colIndex)) = v;
    }

    /**
     * @brief: Returns the dimensions (M, N, LDA) of the Matrix as tuple
     */
    tuple<int, int, int> getDimensions() {
        return {M, N, LDA};
    }

    /**
     * @brief: Returns pointer to data, used for cleanup in main
     */
    auto getData() {
        return data;
    }

    /**
     * @brief: Returns the sum of 2 matrices, assuming the 2 matrices have the same dimensions M and N
     * @param other: The other matrix being added to this matrix
     * Note: the matrices may have different LDA's
     * 
     * @return: The matrix sum
     */
    Matrix matrixAddition(Matrix& other) {
        // Make variables storing rows, cols, and LDA for both matrices
        int rowsA = get<0>(this->getDimensions());
        int colsA = get<1>(this->getDimensions());
        int rowsB = get<0>(other.getDimensions());
        int colsB = get<1>(other.getDimensions());

        // Preliminary check: ensure A and B have the same number of rows and columns
        if (rowsA != colsA || colsA != colsB) {
            throw runtime_error("Error: For matrix addition, A and B must have the same dimensions");

        }

        // Define the empty matrix of 0's, C
        double* CDataPtr = new double[rowsA*colsB];
        // Number of rows and cols should match those of A and B
        // Don't need to embed within larger matrix, so make LDA equal to the number of rows
        Matrix C = Matrix(rowsA, colsA, rowsA, CDataPtr);

        for (int i = 0; i < rowsA; ++i) {
            for (int j = 0; j < colsB; ++j) {
                // C[i, j] = A[i, j] + B[i, j]
                C.set(i, j, this->at(i, j) + other.at(i, j));
            }
        }
        return C;
    }

    /**
     * @brief: Provides a std::cout output of the matrix elements
     */
    void toString() {
        for (int r = 0; r < M; ++r) {
            for (int c = 0; c < N; ++c) {
                cout << at(r, c) << " ";
            }
            cout << "\n";
        }
        cout << "\n";
    }

    /**
     * @brief: Returns the top left quarter of the matrix (i.e. A11) - used in getSubmatrices()
     */
    Matrix TopLeft(int i, int j) {
        return Matrix(i, j, LDA, data.data_handle(), topLeftIndex);
    }

    /**
     * @brief: Returns the top right quarter of the matrix (i.e. A12) - used in getSubmatrices()
     */
    Matrix TopRight(int i, int j) {
        return Matrix(i, j, LDA, data.data_handle(), topLeftIndex + j*LDA);
    }

    /**
     * @brief: Returns the bottom left quarter of the matrix (i.e. A21) - used in getSubmatrices()
     */
    Matrix BotLeft(int i, int j) {
        return Matrix(i, j, LDA, data.data_handle(), topLeftIndex + i);
    }

    /**
     * @brief: Returns the bottom right quarter of the matrix (i.e. A22) - used in getSubmatrices()
     */
    Matrix BotRight(int i, int j) {
        return Matrix(i, j, LDA, data.data_handle(), topLeftIndex + j*LDA + i);
    }

    /**
     * @brief: Returns a tuple of the 4 submatrices of the current matrix
     * These are the top left (A11), top right (A12), bottom left (A21), and bottom right (A22) submatrices
     */
    tuple<Matrix, Matrix, Matrix, Matrix>  getSubmatrices(){
        // Use M/2 and N/2 since splitting dimensions in half, in order to get the quarter submatrices
        return {TopLeft(M/2, N/2), TopRight(M/2, N/2), BotLeft(M/2, N/2), BotRight(M/2, N/2)};
    }

    /** Naive Matrix Multiplication
     * @brief: When the blocked matrix is small enough to fit in the cache, executes the matrix product naively
     * @param other: The other matrix (B)
     * @param out: The product matrix (C)
     * @param dim: The dimension of the smallest block matrix (usually 2 since we compute 2x2 matrix products naively)
     */
    void MatMult(Matrix& other, Matrix& out, int dim) {
        for(int i = 0; i < dim; i++) { 
            for(int j = 0; j < dim; j++) {
                for(int k = 0; k < dim; k++){ 
                    // C[i, j] += A[i, k] * B[k, j]
                    out.set(i, j, out.at(i, j) + this->at(i, k) * other.at(k, j));
                }
            }
        }
    }

    /** Recursive Block Multiplication 
     * @brief: Calculate the product of 2 square matrices of dimension 2^n using a recursive block formula
     * 
     * @param other: The other matrix (B)
     * @param out: The product matrix (C)
     * @param dim: The dimension of the current block matrix
     */
    void RecursiveMatMult(Matrix& other, Matrix& out, int dim){
        /** The recursive formula start off at the largest block and recurses down to blocks of 4 elements:
        * If the dimension of the block is greater than 2 (there are more than 4 elements in a block)
        *   then recursively find all 8 sections of that block 
        * (EXAMPLE)
        * A = A11 A12,   B = B11 B12,   C = C11 C12 
        *     A21 A22        B21 B22        C21 C22
        * Where A * B = C
        * Note: this is equivalent to 
        *   C11 = A11 * B11 + A12 * B21
        *   C12 = A11 * B12 + A12 * B22
        *   C21 = A21 * B11 + A22 * B21
        *   C22 = A21 * B12 + A22 * B22
        * First call finds A11 * B11. If ther are more than 4 elements then split into 4 submatrices 
        * A11 = (A11)11 (A11)12,  B11 = (B11)11 (B11)12
        *       (A11)21 (A11)22         (B11)21 (B11)22
        * Keep splitting until 4 elements in submatrix (2x2), then use Naive Matrix Multiplication and add the product to C
        * Finish all 8 calls of A11 * B11 (call 4 is done above) (including all recursions "below" the block) 
        * Then finish all 8 calls of A*B (including recursions below all blocks)
        * Add to C to complete entire matrix calculation
        */
        if (dim > 2){
            // Divide dimension in half, cut block into 4ths
            dim /= 2;

            auto [tla, tra, bla, bra] = this->getSubmatrices();  // Submatrices of A (A11, A12, A21, A22)
            auto [tlb, trb, blb, brb] = other.getSubmatrices();  // Submatrices of B (B11, B12, B21, B22)
            auto [tlc, trc, blc, brc] = out.getSubmatrices();    // Submatrices of C (C11, C12, C21, C22)
            

            // C11 = (A11 * B11) + (A12 * B21)
            tla.RecursiveMatMult(tlb, tlc, dim); // C11 += A11 * B11
            tra.RecursiveMatMult(blb, tlc, dim); // C11 += A12 * B21

            // C12 = (A11 * B12) + (A12 * B22)
            tla.RecursiveMatMult(trb, trc, dim); // C12 += A11 * B12
            tra.RecursiveMatMult(brb, trc, dim); // C12 += A12 * B22


            // C21 = (A21 * B11) + (A22 * B21)
            bla.RecursiveMatMult(tlb, blc, dim); // C21 += A21 * B11
            bra.RecursiveMatMult(blb, blc, dim); // C21 += A22 * B21


            // C22 = (A21 * B12) + (A22 * B22)
            bla.RecursiveMatMult(trb, brc, dim); // C22 += A21 * B12
            bra.RecursiveMatMult(brb, brc, dim); // C22 += A22 * B22

            
        }
        else { 
            // Base Case: Once matrix dimension is small enough (i.e. 2x2), compute product naively
            this->MatMult(other, out, dim);
        }
    }

    /** Starter function
     * @brief: Starter function that calls recursive block function
     * @param other: The other matrix (B)
     * 
     * @return C: The matrix product of A*B, (C)
     */
    Matrix BlockedMatMult(Matrix& other, bool parallel) {
        auto [M, N, LDA] = this->getDimensions();
        // Initialize empty matrix C
        double* CDataPtr = new double[M*N];
        fill_n(CDataPtr, M * N, 0.0);  // Initialize all values to 0
        Matrix C = Matrix(M, N, LDA, CDataPtr);

        auto [tla, tra, bla, bra] = this->getSubmatrices();  // Submatrices of A (A11, A12, A21, A22)
        auto [tlb, trb, blb, brb] = other.getSubmatrices();  // Submatrices of B (B11, B12, B21, B22)
        auto [tlc, trc, blc, brc] = C.getSubmatrices();    // Submatrices of C (C11, C12, C21, C22)
        int dim = M/2;
        if (parallel) {
            // Create parallel region for the 4 threads to execute the following recursions in parallel
            #pragma omp parallel 
            {
                // Ensure blocks of code executed by one thread
                #pragma omp single 
                {
                    // Create tasks executed by any thread, enabling parallelism
                    #pragma omp task 
                    {
                        // C11 = (A11 * B11) + (A12 * B21)
                        tla.RecursiveMatMult(tlb, tlc, dim); // C11 += A11 * B11
                        tra.RecursiveMatMult(blb, tlc, dim); // C11 += A12 * B21
                    }
                    #pragma omp task 
                    {
                        // C12 = (A11 * B12) + (A12 * B22)
                        tla.RecursiveMatMult(trb, trc, dim); // C12 += A11 * B12
                        tra.RecursiveMatMult(brb, trc, dim); // C12 += A12 * B22
                    }
                    #pragma omp task 
                    {
                        // C21 = (A21 * B11) + (A22 * B21)
                        bla.RecursiveMatMult(tlb, blc, dim); // C21 += A21 * B11
                        bra.RecursiveMatMult(blb, blc, dim); // C21 += A22 * B21
                    }
                    #pragma omp task 
                    {
                        // C22 = (A21 * B12) + (A22 * B22)
                        bla.RecursiveMatMult(trb, brc, dim); // C22 += A21 * B12
                        bra.RecursiveMatMult(brb, brc, dim); // C22 += A22 * B22
                    }
                }
            }
            #pragma omp taskwait
        }
        else {
            // C11 = (A11 * B11) + (A12 * B21)
            tla.RecursiveMatMult(tlb, tlc, dim); // C11 += A11 * B11
            tra.RecursiveMatMult(blb, tlc, dim); // C11 += A12 * B21
            // C12 = (A11 * B12) + (A12 * B22)
            tla.RecursiveMatMult(trb, trc, dim); // C12 += A11 * B12
            tra.RecursiveMatMult(brb, trc, dim); // C12 += A12 * B22
            // C21 = (A21 * B11) + (A22 * B21)
            bla.RecursiveMatMult(tlb, blc, dim); // C21 += A21 * B11
            bra.RecursiveMatMult(blb, blc, dim); // C21 += A22 * B21
            // C22 = (A21 * B12) + (A22 * B22)
            bla.RecursiveMatMult(trb, brc, dim); // C22 += A21 * B12
            bra.RecursiveMatMult(brb, brc, dim); // C22 += A22 * B22
        }
        return C;
    }

    /**
     * @brief: Naive matrix multiplication, used for comparing speed of Naive and Recursive Block Matrix Multiplication
     * @param other: The other matrix (B)
     * 
     * @return C: The matrix product of A*B, (C), computed naively
     */
    Matrix NaiveMatMult(Matrix& other) {
        auto [A_M, A_N, A_LDA] = this->getDimensions();
        auto [B_M, B_N, B_LDA] = other.getDimensions();
        // Initialize empty matrix C
        double* CDataPtr = new double[A_M*B_N];
        fill_n(CDataPtr, A_M * B_N, 0.0);  // Initialize all values to 0
        Matrix C = Matrix(M, N, LDA, CDataPtr);
        
        // Naive matrix multiplication
        for(int i = 0; i < A_M; ++i) {
            for(int j = 0; j < B_N; ++j) {
                int sum = 0;
                for (int k = 0; k < A_N; ++k) {
                    // C[i, j] += A[i, k] * B[k, j]
                    sum += this->at(i, k) * other.at(k, j);
                }
                C.set(i, j, sum);
            }
        }
        return C;
    }

    /**
     * @brief: Checks if this matrix is equal to another one (elements at every location are the same)
     * @param: The other matrix (B)
     * 
     * @return: True if equal, false otherwise
     */
    bool equalMatrices(Matrix& other) {
        auto [M, N, LDA] = getDimensions();
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                if (abs((this->at(i, j) - other.at(i, j)) / this->at(i, j)) > 0.01) {  // If relative error is greater than 1%
                    cout << "i = " << i << ", j = " << j << endl;
                    return false;  // Matrices not equal if difference in elements is above threshold
                }
            }   
        }
        return true;
    }
};


int main() {
    int n;
    cout << "Welcome user! This program will compute the matrix-matrix product of two matrices of random numbers." << endl;
    cout << "Please enter a integer value n, such that the dimension of the matrix will be 2^n: ";
    cin >> n;
    cout << "\nYou entered n = " << n << endl;
    int N = pow(2, n);

    char parallelChoice;
    cout << "Would you like the program to utilize parallelization? (Enter 'y' for yes, 'n' for no): ";
    cin >> parallelChoice;
    bool useParallel = (parallelChoice == 'y');
    if (useParallel)
        cout << "\nYou entered, yes, you would like the computation to use parallelization." << endl;
    else
        cout << "\nYou entered, no, you would not like the computation to use parallelization." << endl;

    cout << "\nComputing the matrix product of A*B = C...\n" << endl;
    
    // Specify the engine and distribution.
    random_device rnd_device;  // First create an instance of an engine.
    mt19937 mersenne_engine {rnd_device()};  // Generates random doubles
    uniform_real_distribution<double> dist {1, 100};  // Random numbers from 1 through 100
    auto gen = [&]() { return dist(mersenne_engine); };

    // Create Matrices A and B to multiply
    double* AData = new double[N * N];
    double* BData = new double[N * N];
    generate(AData, AData + (N * N), gen);
    generate(BData, BData + (N * N), gen);
    Matrix A = Matrix(N, N, N, AData);
    Matrix B = Matrix(N, N, N, BData);
    
    // Test duration for Recursive Blocked Matrix Multiplication
    using clock = chrono::steady_clock;  // Use steady clock for precise measurements
    clock::time_point before = clock::now();
    Matrix C = A.BlockedMatMult(B, useParallel); 
    auto after = clock::now();
    auto time = static_cast<double>(chrono::duration_cast<chrono::milliseconds> (after-before).count());
    cout << "Recursive Duration: " << chrono::duration_cast<chrono::milliseconds> (after-before).count() << " ms" << endl;
    double numOps = 2.0 * pow(N, 3);
    auto secs = static_cast<double>(chrono::duration_cast<chrono::milliseconds> (after-before).count()) / 1000.;
    double FLOPS = numOps / secs;
    cout << "Recursive FLOPS = " << FLOPS << ", GFLOPS = " << FLOPS/(1.e9) << "\n" << endl;
    
    // Test duration for Naive Matrix Multiplication
    before = clock::now();
    Matrix D = A.NaiveMatMult(B); 
    after = clock::now();
    cout << "Naive Duration: " << chrono::duration_cast<chrono::milliseconds> (after-before).count() << " ms" << endl;
    secs = static_cast<double>(chrono::duration_cast<chrono::milliseconds> (after-before).count()) / 1000.;
    FLOPS = numOps / secs;
    cout << "Naive FLOPS = " << FLOPS << ", GFLOPS = " << FLOPS/(1.e9) << "\n" << endl;
    

    // Check that recursive result matches naive result    
    if (C.equalMatrices(D)) {
        cout << "Recursive and Naive approach computes same product. Matrix product results match!" << endl;
    } else {
        cout << "Error: Matrix product results do NOT match" << endl;
    }

    // Cleanup
    delete[] AData;
    delete[] BData;
    delete[] C.getData().data_handle();
    delete[] D.getData().data_handle();

    return 0;    
}
