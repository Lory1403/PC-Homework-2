# PC-Homework-2
Introduction to Parallel Computing - Homework 2: Parallelizing matrix operations using OpenMP

This repository contains the solutions developed for the second assignment of the course ***"Introduction to Parallel Computing [146209] - Prof. Vella"*** about ***Parallelizing matrix operations using OpenMP***.

## Execution Guide
This guide provides instructions on how to set up and execute the provided C code on the High-Performance Computing (HPC) cluster and locally.

- [Prerequisites](#prerequisites)
- [HPC Cluster](#execution)
- [Locally](#usage)
- [Results](#example-usages)

## Prerequisites
> [!IMPORTANT] 
> The code requires a GCC compile (C++) with OpenMP support.

## Compile and run
### HPC Cluster
Clone the repository, modify the PBS file to point to the right directories (line 30) and submit it to the queue with the command qsub `omp_hw2.pbs`.\
Normal outputs are not accessible during processing and a dump will be located in `hw2.o` and `hw2.e` files after the execution is complete.

### Locally
For manual execution, compile both source files:
```shell
g++ -fopenmp hw1_pt1.c -o hw1_pt1.out 
g++ -fopenmp hw1_pt2.c -o hw1_pt2.out
```
Run the compiled programs:
```shell
./hw1_pt1.out
./hw1_pt2.out
```
## Results
Results are stored in .txt files in the folders `results_hw2_pt1/` and `results_hw2_pt2/`.

The report analyzing the results is available [here](report/build/report.pdf).