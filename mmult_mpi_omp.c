#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/times.h>
#define min(x, y) ((x)<(y)?(x):(y))

double* gen_matrix(int n, int m);
int mmult(double *c, double *a, int aRows, int aCols, double *b, int bRows, int bCols);
void compare_matrix(double *a, double *b, int nRows, int nCols);

/** 
    Program to multiply a matrix times a matrix using both
    mpi to distribute the computation among nodes and omp
    to distribute the computation among threads.
*/

int main(int argc, char* argv[])
{
  int nrows, ncols;
  double *aa;	/* the A matrix */
  double *bb;	/* the B matrix */
  double *cc1;	/* A x B computed using the omp-mpi code you write */
  double *cc2;	/* A x B computed using the conventional algorithm */

  int a_row, a_col, b_col;

  int myid, numprocs;
  double starttime, endtime;
  MPI_Status status;
  /* insert other global variables here */
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if (argc > 1) {

    a_row = atoi(argv[1]);
    a_col = atoi(argv[2]);
    b_col = atoi(argv[3]);

    printf("The matrices' dimensions are %d x %d and %d x %d.\n", a_row, a_col, a_col, b_col);

    // testing if the matrices are generated correctly
    aa = gen_matrix(a_row, a_col);
    bb = gen_matrix(a_col, b_col);

    printf("Matrix A:\n");
    for (int i = 0; i < a_row; i++) {
      for (int j = 0; j < a_col; j++) {
        printf("%d ", aa[i * a_row + j]);
      }
      printf("\n");
    }

    printf("Matrix B:\n");
    for (int i = 0; i < a_col; i++) {
      for (int j = 0; j < b_col; j++) {
        printf("%d ", bb[i * a_col + j]);
      }
      printf("\n");
    }

    /**
    if (myid == 0) {
      // Master Code goes here
      aa = gen_matrix(nrows, ncols);
      bb = gen_matrix(ncols, nrows);
      cc1 = malloc(sizeof(double) * nrows * nrows); 
      starttime = MPI_Wtime();

      // Insert your master code here to store the product into cc1
      endtime = MPI_Wtime();
      printf("%f\n",(endtime - starttime));
      cc2  = malloc(sizeof(double) * nrows * nrows);
      mmult(cc2, aa, nrows, ncols, bb, ncols, nrows);
      compare_matrices(cc2, cc1, nrows, nrows);
    } else {
      // Slave Code goes here
    }
  } else {
    fprintf(stderr, "Usage matrix_times_vector <size>\n");
  }
  MPI_Finalize();
  */
  }
  return 0;
}
