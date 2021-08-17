#include <stdlib.h>
#include <iostream>

using namespace std;

void matmul(double *a, double *b, double *res, int m, int n, int p) {
    /**
     * @brief
     *     Compute the matrix multiplication of a and b, store the result in c
     *
     * @note
     *     1. Matrices are represented as 1-d arrays
     *     2. NO DIMENSION CHECKING!!! So make sure that a and b are multiplicable
     *
     * @para:
     *     a: First input matrix, has dimension (m, n)
     *     b: Second input matrix, has dimension (n, p)
     *     res: output matrix, has dimension (m, p)
     */
    
    for (int row = 0; row < m; row++) {
        
        for (int col = 0; col < p; col++) {
            
            // compute element (row, col) in matrix `res`
            int ans = 0;
            
            for (int k = 0; k < n; k++) {
                
                // the row'th row in A
                int a_idx = row * n + k;
                
                // the col'th column in B
                int b_idx = k * p + col;
                
                // dot product
                ans += a[a_idx] * b[b_idx];
            }
            
            // store the result
            res[row * p + col] = ans;
        }
    }    
}

void print_matrix(double *matrix, int m, int n) {
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) cout << matrix[row * n + col] << " ";
        cout << endl;
    }
}

int main() {
    
    double *input_left, *input_right, *output;
    int m = 2, n = 3, p = 2;

    input_left = (double*)malloc(m * n * sizeof(double));
    input_right = (double*)malloc(n * p * sizeof(double));
    output = (double*)malloc(m * p * sizeof(double));
    
    for (int i = 0; i < m * n; i++) input_left[i] = 1.0;
    for (int i = 0; i < n * p; i++) input_right[i] = 2.0;

    cout << "The input matrix A:" << endl;
    print_matrix(input_left, m, n);
    cout << "The input matrix B:" << endl;
    print_matrix(input_right, n, p);

    matmul(input_left, input_right, output, m, n, p);

    cout << "The output matrix:" << endl;
    print_matrix(output, m, p);

    free(input_left);
    free(input_right);
    free(output);
    return 0;
}