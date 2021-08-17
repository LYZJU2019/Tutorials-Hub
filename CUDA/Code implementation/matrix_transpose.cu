#include <iostream>

#define BLOCK_SIZE 8

using namespace std;

__global__ void matirx_transpose(double *input, double *output, int m, int n) {
    
    // which element (row, col) in the output matrix does this thread operate?
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // calculate the index of 1-d representation
    // output(j, i) = input(i, j)
    int output_idx = row * m + col;
    int input_idx = col * n + row;
    
    // write if the index does not exceed the boundary
    if (row < n && col < m)
    	output[output_idx] = input[input_idx];
}

void print_matrix(double *matrix, int m, int n) {
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) cout << matrix[row * n + col] << " ";
        cout << endl;
    }
}

int main() {
    // row and column
    int m = 17, n = 9;

    // allocate space for input and output matrix in CPU
    double *h_input, *h_output;
    h_input = (double*)malloc(m * n * sizeof(double));
    h_output = (double*)malloc(n * m * sizeof(double));

    // initialize input matrix
    for (int idx = 0; idx < m * n; idx++) h_input[idx] = 1.0;
    cout << "The input matrix:" << endl;
    print_matrix(h_input, m, n);

    // allocate space for input and output matrix in GPU
    double *d_input, *d_output;
    cudaMalloc((void**)&d_input, m * n * sizeof(double));
    cudaMalloc((void**)&d_output, n * m * sizeof(double));

    // copy the content of the input matrix to GPU
    cudaMemcpy(d_input, h_input, m * n * sizeof(double), cudaMemcpyHostToDevice);
    
    // set configuration for grid and thread blocks
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((m + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    matirx_transpose<<<grid, block>>>(d_input, d_output, m, n);

    // copy the result back to CPU
    cudaMemcpy(h_output, d_output, n * m * sizeof(double), cudaMemcpyDeviceToHost);

    // check the result
    cout << "The matrix after transpose:" << endl;
    print_matrix(h_output, n, m);
    
    // free memories in CPU and GPU
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}