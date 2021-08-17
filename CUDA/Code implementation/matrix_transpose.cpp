#include <stdlib.h>
#include <iostream>

using namespace std;

void matrix_transpose(double *input, double *output, int m, int n) {
    /**
     * @brief:
     *      transpose the matrix `input` and store the result in `output`
     *
     * @para:
     *     `input` represents a matrix with m rows and n columns.
     *
     *     `output` represents a matrix with n rows and m columns,
	 *      which is the transpose of matrix `input`
	 */

    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            output[col * m + row] = input[row * n + col];
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
    
    double *input, *output;
    int m = 1, n = 10;

    input = (double*)malloc(m * n * sizeof(double));
    output = (double*)malloc(m * n * sizeof(double));
    
    for (int i = 0; i < m * n; i++) input[i] = 1.0;

    cout << "matrix before transpose:" << endl;
    print_matrix(input, m, n);

    matrix_transpose(input, output, m, n);

    cout << "matrix after transpose:" << endl;
    print_matrix(output, n, m);

    free(input);
    free(output);
    return 0;
}