#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <limits.h>
#ifdef _OPENMP
    #include <omp.h>
#endif

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

void sumMatrix(float **matrix, int rows, int cols)
{
    omp_set_num_threads(maxThreads);

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

void displayMatrix(float **matrix, int rows, int cols)
{
    // Display the results
	for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) 
            printf("%f,", matrix[i][j]);
		printf("\n");
    }
}

void matT(FILE *file, int dimension, float **A, float **TA, int rows, int cols)
{
    double start_time = clock();

    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < rows; j++) {
            TA[i][j] = A[j][i];
        }
    }

    double end_time = clock();
    double elapsed_time = (end_time - start_time) / CLOCKS_PER_SEC;
    printf("Mat T\tElapsed time: %f\n", elapsed_time);

    fprintf(file, "%d;1;%f;serial\n", dimension, elapsed_time);
}

void matBlockT(FILE *file, int dimension, int block_size, float **A, float **TA, int rows, int cols)
{
    int num_row_blocks = rows / block_size;
    int num_col_blocks = cols / block_size;

    // 3double start_time = clock();

    // for (int i = 0; i < num_col_blocks; i++) {
    //     for (int j = 0; j < num_row_blocks; j++) {
    //         for (int bi = 0; bi < block_size; bi++) {
    //             for (int bj = 0; bj < block_size; bj++) {
    //                 TA[i * block_size + bi][j * block_size + bj] = A[j * block_size + bj][i * block_size + bi];    
    //             }
    //         }
    //     }
    // }

    // double end_time = clock();
    // double elapsed_time = (end_time - start_time) / CLOCKS_PER_SEC;
    // printf("Mat Block T - 1\tElapsed time: %f\n", elapsed_time);
    // sumMatrix(TA, cols, rows);

    // fprintf(file, "%d;1;%f;%d;serial1\n", dimension, elapsed_time, block_size);

    double start_time = clock();
    for (int i = 0; i < rows; i += block_size) {
        for (int j = 0; j < cols; j += block_size) {
            for (int bi = 0; bi < block_size; bi++) {
                for (int bj = 0; bj < block_size; bj++) {
                    TA[j + bj][i + bi] = A[i + bi][j + bj];
                }
            }
        }
    }
    double end_time = clock();
    double elapsed_time = (end_time - start_time) / CLOCKS_PER_SEC;
    printf("Mat Block T\tElapsed time: %f\n", elapsed_time);
    sumMatrix(TA, cols, rows);

    fprintf(file, "%d;1;%f;%d;serial\n", dimension, elapsed_time, block_size);

    // 2double start_time = clock();
    // for (int i = 0; i < cols; i += block_size) {
    //     for (int j = 0; j < rows; j += block_size) {
    //         for (int bi = 0; bi < block_size; bi++) {
    //             for (int bj = 0; bj < block_size; bj++) {
    //                 TA[i+ bi][j + bj] = A[j + bj][i + bi];
    //             }
    //         }
    //     }
    // }
    // double end_time = clock();
    // double elapsed_time = (end_time - start_time) / CLOCKS_PER_SEC;
    // printf("Mat Block T - 3\tElapsed time: %f\n", elapsed_time);
    // sumMatrix(TA, cols, rows);

    // fprintf(file, "%d;1;%f;%d;serial3\n", dimension, elapsed_time, block_size);
}

void matTpar(FILE *file, int dimension, int num_threads, float **A, float **TA, int rows, int cols)
{
    omp_set_num_threads(num_threads);

    double start_time = omp_get_wtime();
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < rows; j++) {
            TA[i][j] = A[j][i];
        }
    }
    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;
    printf("Mat T Par\tElapsed time: %f\n", elapsed_time);
    sumMatrix(TA, cols, rows);

    fprintf(file, "%d;%d;%f;parallel\n", dimension, num_threads, elapsed_time);
}

void matBlockTpar(FILE *file, int dimension, int num_threads, int block_size, float **A, float **TA, int rows, int cols)
{
    omp_set_num_threads(num_threads);

    double start_time = omp_get_wtime();
    #pragma omp parallel for simd collapse(4)
    for (int i = 0; i < rows; i += block_size) {
        for (int j = 0; j < cols; j += block_size) {
            for (int bi = 0; bi < block_size; bi++) {
                for (int bj = 0; bj < block_size; bj++) {
                    TA[j + bj][i + bi] = A[i + bi][j + bj];
                }
            }
        }
    }
    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;
    printf("Mat Block T Par\tElapsed time: %f\n", elapsed_time);
    sumMatrix(TA, cols, rows);

    fprintf(file, "%d;%d;%f;%d;parallel\n", dimension, num_threads, elapsed_time, block_size);
}

int compare_integers(const void* a, const void* b) {
    return (*(int*)a - *(int*)b);
}

int* findHalfDivisors(int rows, int cols) {
    int smaller = (rows < cols) ? rows : cols;

    int* common_divisors = (int*)malloc(smaller * sizeof(int));

    int common_count = 0;

    for (int i = 2; i * i <= smaller; i++) {
        if (rows % i == 0 && cols % i == 0) {
            common_divisors[common_count++] = i;

            if (i * i != smaller) {
                int other_divisor = smaller / i;
                if (rows % other_divisor == 0 && cols % other_divisor == 0) {
                    common_divisors[common_count++] = other_divisor;
                }
            }
        }
    }

    qsort(common_divisors, common_count, sizeof(int), compare_integers);

    // Allocazione di memoria per la nuova metà dell'array
    int* halfDivisors = (int*)malloc((common_count/2 + 1) * sizeof(int));

    // Copiare la prima metà dell'array originale (si può copiare anche la seconda metà)
    for (int i = 0; i < common_count/2; i++) {
        halfDivisors[i] = common_divisors[i];
    }
    // Mark the end of the array with a sentinel value (-1)
    halfDivisors[common_count/2] = -1;
    
    free(common_divisors);
    return halfDivisors;
}

int main(int argc, char const *argv[])
{
#ifndef _OPENMP
    perror("OpenMP is required to run this program.\n");
    exit(1);
#endif

    srand(time(NULL));

    #pragma omp parallel
    {
        #pragma omp master
        {
            maxThreads = omp_get_max_threads();
        }
    }

    const char* folder = "dense_results_hw2_pt2"; // Sostituisci con il nome della cartella desiderato

    if (access(folder, F_OK) == -1) {
        if (mkdir(folder, 0777) != 0) {
            perror("Errore nella creazione della cartella.\n");
            exit(1);
        }
    }

    FILE *file_transposed = fopen("dense_results_hw2_pt2/results_hw2_pt2_transposed.txt", "w");
    if (file_transposed == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }

    FILE *file_transposed_block = fopen("dense_results_hw2_pt2/results_hw2_pt2_transposed_block.txt", "w");
    if (file_transposed_block == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }

    fprintf(file_transposed, "Dimension;Threads;Elapsed time;Method\n");
    fprintf(file_transposed_block, "Dimension;Threads_num;Elapsed time;Block size,Method\n");

    float **A, **TA;
    int rows, cols;
    bool isSparse = false;
    float sparsity = ((float)rand() / RAND_MAX) * (1 - 0.8) + 0.8;

    int maxDimension = 80000;
    for(int dimension = 1250; dimension <= maxDimension; dimension*=4) {
        rows = cols = dimension;

        // Matrix allocation
        A = matrixAllocation(rows, cols);

        if (isSparse) {
            // Initialize sparse matrix A
            sparseMatricesInitialization(A, rows, cols, sparsity);
        } else {
            // Initialize dense matrix A
            denseMatricesInitialization(A, rows, cols);
        }

        TA = matrixAllocation(cols, rows);

        matT(file_transposed, dimension, A, TA, rows, cols);

        for (int threads = 2; threads <= maxThreads; threads*=2) {
            // Reset matrix
            matrixDeallocation(TA, rows, cols);
            TA = matrixAllocation(cols, rows);

            matTpar(file_transposed, dimension, threads, A, TA, rows, cols);
        }

        // Cambia il codice per i blocchi
        int* common_divisors = findHalfDivisors(rows, cols);

        printf("Common divisors: ");
        for (int i = 0; common_divisors[i] != -1; i++) {
            // Reset matrix
            matrixDeallocation(TA, rows, cols);
            TA = matrixAllocation(cols, rows);

            matBlockT(file_transposed_block, dimension, common_divisors[i], A, TA, rows, cols);

            for (int threads = 2; threads <= maxThreads; threads*=2) {
                // Reset matrix
                matrixDeallocation(TA, rows, cols);
                TA = matrixAllocation(cols, rows);

                matBlockTpar(file_transposed_block, dimension, threads, common_divisors[i], A, TA, rows, cols);
            }
        }
    
        free(common_divisors);
        
        // Free dynamically allocated memory
        matrixDeallocation(A, rows, cols);
        matrixDeallocation(TA, cols, rows);
    }

    fclose(file_transposed);
    fclose(file_transposed_block);

    return 0;
}
