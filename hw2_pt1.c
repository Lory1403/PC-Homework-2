#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#ifdef _OPENMP
    #include <omp.h>
#endif
#include <limits.h>

int maxThreads;

void printProgressBar(int progress, int total)
{
    int barWidth = 50;
    float percentage = (float)progress / total;
    int pos = (int)(barWidth * percentage);

    printf("[");
    for (int i = 0; i < barWidth; i++) {
        if (i < pos) {
            printf("=");
        } else if (i == pos) {
            printf(">");
        } else {
            printf(" ");
        }
    }

    printf("] %3.2f%%\r", percentage * 100.0);
    fflush(stdout);
}

float **matrixAllocation(int rows, int columns)
{
    omp_set_num_threads(maxThreads);

    // Dynamically allocate memory for matrices A, B, and C
    float **matrix = (float**)malloc(rows * sizeof(float*));
    #pragma omp parallel for shared(matrix)
    for (int i = 0; i < rows; i++) {
        matrix[i] = (float*)malloc(columns * sizeof(float));
    }
    return matrix;
}

void matrixDeallocation(float **matrix, int rows, int columns)
{
    omp_set_num_threads(maxThreads);

    #pragma omp parallel for shared(matrix)
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

// Initialize matrices A, B, and C
void denseMatricesInitialization(float **matrix, int rows, int cols)
{
    omp_set_num_threads(maxThreads);

    #pragma omp parallel for shared(matrix) collapse(2)
    for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			matrix[i][j] = (float)rand() / RAND_MAX;
}

void sparseMatricesInitialization(float **matrix, int rows, int cols, float sparsity)
{
    omp_set_num_threads(maxThreads);

    #pragma omp parallel for shared(matrix) collapse(2)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // Generate a random value between 0 and 1
            float randValue = (float)rand() / RAND_MAX;

            // Check if the random value is less than the desired sparsity
            if (randValue > sparsity) {
                // If true, set the matrix element to a non-zero value
                matrix[i][j] = randValue;
            } else {
                // If false, set the matrix element to zero
                matrix[i][j] = 0.0;
            }
        }
    }
}

// Testing the correctness of the algorithm
void sumMatrix(float **matrix, int rows, int cols)
{
    // Matrix addition
    long int sum = 0;
    #pragma omp parallel for shared(matrix) reduction(+:sum) collapse(2)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            sum += matrix[i][j];
        }
    }
    printf("Sum of matrix C: %ld\n", sum);
}

void printResults(double elapsed_time, double flops, float **C, int rowsA, int colsB)
{
    printf("Elapsed time: %f seconds\n", elapsed_time);
    printf("GFLOPS: %f\n", flops);
    sumMatrix(C, rowsA, colsB);
}

void saveResults(FILE *file, int dimension, int threads_num, double elapsed_time, double flops)
{
    fprintf(file, "\n%d;%d;%f;%f", dimension, threads_num, elapsed_time, flops);
}

// Display the results
void displayResults(float **A, float **B, float **C, int rowsA, int colsA, int rowsB, int colsB)
{
    printf("Matrix C = A * B:\n");
	printf("A = \t");
    #pragma omp parallel for shared(A) ordered
	for (int i = 0; i < rowsA; i++) {
        #pragma omp parallel for ordered
        for (int j = 0; j < colsA; j++) {
            #pragma omp ordered 
            printf("%f\t", A[i][j]);
        }
        #pragma omp ordered
        printf("\n\t");
    }

	printf("\nB =\t");
    #pragma omp parallel for shared(B) ordered
	for (int i = 0; i < rowsB; i++) {
        #pragma omp parallel for ordered
        for (int j = 0; j < colsB; j++) {
            #pragma omp ordered 
            printf("%f\t", B[i][j]);
        }
        #pragma omp ordered
		printf("\n\t");
    }

	printf("\nC =\t");
    #pragma omp parallel for shared(C) ordered
    for (int i = 0; i < rowsA; i++) {
        #pragma omp parallel for ordered
        for (int j = 0; j < colsB; j++) {
            #pragma omp ordered
            printf("%f\t", C[i][j]);
        }
        #pragma omp ordered
        printf("\n\t");
    }
}

void matMul(FILE *file, int dimension, float **A, float **B, float **C, int rowsA, int colsA, int rowsB, int colsB)
{
    double start_time = clock();

    // Matrix multiplication		
	for (int i = 0; i < rowsA; i++) {
        for (int k = 0; k < rowsB; k++) {
            for (int j = 0; j < colsB; j++) {
                if (k == 0) {
                    C[i][j] = 0.0; // Initialize C[i][j] only when k is 0
                }
                C[i][j] += A[i][k] * B[k][j];
            }
        }
        // printProgressBar(i, rowsA);
    }

    double end_time = clock();
    double elapsed_time = (end_time - start_time) / CLOCKS_PER_SEC;
    printf("elapsed time: %f", elapsed_time);
    long long int operations = 2 * (long long int) rowsA * (long long int) colsB * (long long int) colsA;
    printf("operations: %lld", operations);
    double flops = (operations / elapsed_time) / 1e9;  // Calculate GFLOPS
    printf("flops: %f", flops);

    // printResults();
    saveResults(file, dimension, 1, elapsed_time, flops);
}

void matMulPar(FILE *file, int dimension, int threads_num, float **A, float **B, float **C, int rowsA, int colsA, int rowsB, int colsB)
{
    omp_set_num_threads(threads_num);
    double start_time = omp_get_wtime();
    float sum;
    #pragma omp parallel for shared(A, B, C) reduction(+:sum) collapse(2)
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            sum = 0.0;
            #pragma omp simd 
            for (int k = 0; k < colsA; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
        // printProgressBar(i, rowsA);
    }

    double end_time = omp_get_wtime();
    double elapsed_time = (end_time - start_time);
    printf("elapsed time: %f", elapsed_time);
    long long int operations = 2 * (long long int) rowsA * (long long int) colsB * (long long int) colsA;
    printf("operations: %ld", operations);
    double flops = (operations / elapsed_time) / 1e9;  // Calculate GFLOPS
    printf("flops: %f", flops);

    // printResults();
    saveResults(file, dimension, threads_num, elapsed_time, flops);
}

int main(int argc, char const *argv[])
{
#ifndef _OPENMP
    perror("OpenMP is required to run this program.\n");
    exit(1);
#endif

    srand(time(NULL));
    
    const char* folder = "results_hw2_pt1"; // Sostituisci con il nome della cartella desiderato

    if (access(folder, F_OK) == -1) {
        if (mkdir(folder, 0777) != 0) {
            perror("Errore nella creazione della cartella.\n");
            exit(1);
        }
    }

    FILE *file = fopen("results_hw2_pt1/results_hw2_pt1.txt", "w");
    if (file == NULL) {
        perror("Error opening file!\n");
        exit(1);
    }

    #pragma omp parallel
    {
        #pragma omp master
        {
            maxThreads = omp_get_max_threads();
        }
    }

    fprintf(file, "Dimension;Threads_num;Elapsed_time;GFLOPS");

    float **A, **B, **C;
    int rowsA, colsA, rowsB, colsB;
    bool isSparse = false;
    float sparsity = ((float)rand() / RAND_MAX) * (1 - 0.8) + 0.8;

    int maxDimension = 4000;
    for(int dimension = 500; dimension <= maxDimension; dimension*=2) {
        rowsA = colsA = rowsB = colsB = dimension;

        // Check if multiplication is possible
        if (colsA != rowsB) {
            perror("The matrices cannot be multiplied with each other.\n");
            exit(1);
        }

        A = matrixAllocation(rowsA, colsA);
        B = matrixAllocation(rowsB, colsB);
        C = matrixAllocation(rowsA, colsB);

        if (isSparse) {
            // Initialize sparse matrices A and B
            sparseMatricesInitialization(A, rowsA, colsA, sparsity);
            sparseMatricesInitialization(B, rowsB, colsB, sparsity);
        } else {
            // Initialize dense matrices A and B
            denseMatricesInitialization(A, rowsA, colsA);
            denseMatricesInitialization(B, rowsB, colsB);
        }

        matMul(file, dimension, A, B, C, rowsA, colsA, rowsB, colsB);

        // Parallel matrix multiplication
        for (int threads_num = 2; threads_num <= maxThreads; threads_num *= 2) {
            matrixDeallocation(C, rowsA, colsB);
            C = matrixAllocation(rowsA, colsB);
            matMulPar(file, dimension, threads_num, A, B, C, rowsA, colsA, rowsB, colsB);
        }
        
        // Free dynamically allocated memory
        matrixDeallocation(A, rowsA, colsA);
        matrixDeallocation(B, rowsB, colsB);
        matrixDeallocation(C, rowsA, colsB);
    }

    // Close the file
    fclose(file);

    return 0;
}
