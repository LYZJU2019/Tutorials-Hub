#include <iostream>

#define BLOCK_SIZE 8

using namespace std;

__global__ void matmul(double *input_left, double *input_right, double *output, int m, int n, int p) {
    
    // get (row, col), the element in the output matrix
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < p) {
        // the row'th row of matrix A and the col'th column of B
        int ans = 0;
        for (int i = 0; i < n; i++) {
            ans += input_left[row * n + i] * input_right[i * p + col];
        }
        output[row * p + col] = ans;
    }
}

void print_matrix(double *matrix, int m, int n) {
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) cout << matrix[row * n + col] << " ";
        cout << endl;
    }
}

int main() {
    // row and column
    int m = 2, n = 3, p = 2;

    // allocate space for input and output matrix in CPU
    double *h_input_left, *h_input_right, *h_output;
    h_input_left = (double*)malloc(m * n * sizeof(double));
    h_input_right = (double*)malloc(n * p * sizeof(double));
    h_output = (double*)malloc(m * p * sizeof(double));

    // initialize input matrix
    for (int idx = 0; idx < m * n; idx++) h_input_left[idx] = 1.0;
    for (int idx = 0; idx < n * p; idx++) h_input_right[idx] = 2.0;
    cout << "The left matrix is:" << endl;
    print_matrix(h_input_left, m, n);
    cout << "The right matrix is:" << endl;
    print_matrix(h_input_right, n, p);

    // allocate space for input and output matrix in GPU
    double *d_input_left, *d_input_right, *d_output;
    cudaMalloc((void**)&d_input_left, m * n * sizeof(double));
    cudaMalloc((void**)&d_input_right, n * p * sizeof(double));
    cudaMalloc((void**)&d_output, m * p * sizeof(double));

    // copy the content of the input matrix to GPU
    cudaMemcpy(d_input_left, h_input_left, m * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_right, h_input_right, n * p * sizeof(double), cudaMemcpyHostToDevice);
    
    // set configuration for grid and thread blocks
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((p + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
    matmul<<<grid, block>>>(d_input_left, d_input_right, d_output, m, n, p);

    // copy the result back to CPU
    cudaMemcpy(h_output, d_output, m * p * sizeof(double), cudaMemcpyDeviceToHost);

    // check the result
    cout << "The output matrix is:" << endl;
    print_matrix(h_output, m, p);
    
    // free memories in CPU and GPU
    free(h_input_left);
    free(h_input_right);
    free(h_output);
    cudaFree(d_input_left);
    cudaFree(d_input_right);
    cudaFree(d_output);
    
    return 0;
}