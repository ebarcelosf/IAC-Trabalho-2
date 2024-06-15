#include "timer.h"
#include <stdio.h>
#include <stdlib.h>
#include "matrix_lib.h"

// Kernel CUDA para multiplicação escalar de matriz
__global__ void scalar_matrix_mult_kernel(float scalar_value, float *matrix, unsigned long int total_elements) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total_elements) {
        matrix[tid] *= scalar_value;
    }
}

// Kernel CUDA para multiplicação de matriz-matriz
__global__ void matrix_matrix_mult_kernel(struct matrix a, struct matrix b, struct matrix c) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Dimensões das submatrizes
    const int TILE_WIDTH = 16;
    const int TILE_HEIGHT = 16;

    // Memória compartilhada para as submatrizes A e B
    __shared__ float As[TILE_HEIGHT][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_HEIGHT];

    // Elemento de saída
    float Cvalue = 0.0;

    // Loop sobre as submatrizes de A e B
    for (int m = 0; m < (a.width + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {
        // Carrega a submatriz As da matriz a para a memória compartilhada
        if (by * TILE_HEIGHT + ty < a.height && m * TILE_WIDTH + tx < a.width) {
            As[ty][tx] = a.rows[(by * TILE_HEIGHT + ty) * a.width + m * TILE_WIDTH + tx];
        } else {
            As[ty][tx] = 0.0;
        }

        // Carrega a submatriz Bs da matriz b para a memória compartilhada
        if (m * TILE_WIDTH + ty < b.width && bx * TILE_WIDTH + tx < b.height) {
            Bs[ty][tx] = b.rows[(m * TILE_WIDTH + ty) * b.width + bx * TILE_WIDTH + tx];
        } else {
            Bs[ty][tx] = 0.0;
        }

        // Sincroniza as threads
        __syncthreads();

        // Calcula o produto escalar parcial
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Cvalue += As[ty][k] * Bs[k][tx];
        }

        // Sincroniza as threads
        __syncthreads();
    }

    // Escreve Cvalue na matriz c
    if (by * TILE_HEIGHT + ty < c.height && bx * TILE_WIDTH + tx < c.width) {
        c.rows[(by * TILE_HEIGHT + ty) * c.width + bx * TILE_WIDTH + tx] = Cvalue;
    }
}

// Função para realizar a multiplicação escalar de matriz na GPU
int scalar_matrix_mult(float scalar_value, struct matrix *matrix) {
    if (matrix == NULL || matrix->rows == NULL) {
        return 0; // Erro: matriz ou seus valores não estão alocados
    }

    unsigned long int total_elements = matrix->height * matrix->width;
    float *d_matrix;

    // Aloca memória na GPU
    cudaMalloc((void **)&d_matrix, total_elements * sizeof(float));

    // Copia os dados da CPU para a GPU
    cudaMemcpy(d_matrix, matrix->rows, total_elements * sizeof(float), cudaMemcpyHostToDevice);

    // Lança o kernel
    unsigned int threadsPerBlock = 256;
    unsigned int numBlocks = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
    scalar_matrix_mult_kernel<<<numBlocks, threadsPerBlock>>>(scalar_value, d_matrix, total_elements);

    // Copia o resultado de volta para a CPU
    cudaMemcpy(matrix->rows, d_matrix, total_elements * sizeof(float), cudaMemcpyDeviceToHost);

    // Libera a memória da GPU
    cudaFree(d_matrix);

    return 1; // Sucesso
}

// Função para realizar a multiplicação de matriz-matriz na GPU
int matrix_matrix_mult(struct matrix *a, struct matrix *b, struct matrix *c) {
    if (a->width != b->height || a->height != c->height || b->width != c->width) {
        return 0; // Verifica se as dimensões das matrizes são compatíveis
    }

    unsigned long int m = a->height;
    unsigned long int n = a->width;
    unsigned long int p = b->width;
    unsigned long int a_size = m * n * sizeof(float);
    unsigned long int b_size = n * p * sizeof(float);
    unsigned long int c_size = m * p * sizeof(float);

    // Aloca memória na GPU
    struct matrix d_a, d_b, d_c;
    cudaMalloc((void **)&(d_a.rows), a_size);
    cudaMalloc((void **)&(d_b.rows), b_size);
    cudaMalloc((void **)&(d_c.rows), c_size);

    // Copia os dados da CPU para a GPU
    cudaMemcpy(d_a.rows, a->rows, a_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b.rows, b->rows, b_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c.rows, c->rows, c_size, cudaMemcpyHostToDevice);

    d_a.height = a->height;
    d_a.width = a->width;
    d_b.height = b->height;
    d_b.width = b->width;
    d_c.height = c->height;
    d_c.width = c->width;

    // Lança o kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((p + threadsPerBlock.x - 1) / threadsPerBlock.x, (m + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrix_matrix_mult_kernel<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c);

    // Copia o resultado de volta para a CPU
    cudaMemcpy(c->rows, d_c.rows, c_size, cudaMemcpyDeviceToHost);

    // Libera a memória da GPU
    cudaFree(d_a.rows);
    cudaFree(d_b.rows);
    cudaFree(d_c.rows);

    return 1; // Sucesso
}