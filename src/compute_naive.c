#include "compute.h"

// Computes the convolution of two matrices
int convolve(matrix_t *a_matrix, matrix_t *b_matrix, matrix_t **output_matrix) {
  
  //get output mat
  *output_matrix = malloc(sizeof(matrix_t));
  int num_output_rows = (a_matrix -> rows) - (b_matrix -> rows) + 1;
  (*output_matrix)->rows = num_output_rows;
  int num_output_cols = (a_matrix -> cols) - (b_matrix -> cols) + 1;
  (*output_matrix)->cols = num_output_cols;
  (*output_matrix)->data = malloc(sizeof(int32_t) * num_output_rows * num_output_cols);
  
  

  //get flip b mat
  matrix_t* flip_b_matrix = malloc(sizeof(matrix_t));
  int flip_b_matrix_cols = b_matrix->cols;
  flip_b_matrix->cols = flip_b_matrix_cols;
  int flip_b_matrix_rows = b_matrix->rows;
  flip_b_matrix->rows = flip_b_matrix_rows;
  
  int flip_b_matrix_size = b_matrix->rows * b_matrix->cols;
  flip_b_matrix->data = malloc(sizeof(int32_t) * flip_b_matrix_size);
  // init flip mat
  for(int i = 0; i < flip_b_matrix_size; i++) {
    flip_b_matrix->data[i] = b_matrix->data[flip_b_matrix_size - 1 - i];
  }

  //convolve
  for (int r = 0; r < num_output_rows; r ++) {
    for (int c = 0; c < num_output_cols; c ++) {
        uint32_t col_a = a_matrix -> cols;
        uint32_t col_b = b_matrix -> cols;
        int sum = 0;
        for (int i = 0; i < flip_b_matrix_size; i++) {
                int cov_row = (i)/(col_b); 
                int j = (cov_row + r)*(col_a);
                sum += a_matrix->data[j + (i%col_b) + c]*flip_b_matrix->data[i];
            }
        (*output_matrix)->data[c + r*num_output_cols] = sum;
      }
  }
  return 0;
}

// Executes a task
int execute_task(task_t *task) {
  matrix_t *a_matrix, *b_matrix, *output_matrix;

  char *a_matrix_path = get_a_matrix_path(task);
  if (read_matrix(a_matrix_path, &a_matrix)) {
    printf("Error reading matrix from %s\n", a_matrix_path);
    return -1;
  }
  free(a_matrix_path);

  char *b_matrix_path = get_b_matrix_path(task);
  if (read_matrix(b_matrix_path, &b_matrix)) {
    printf("Error reading matrix from %s\n", b_matrix_path);
    return -1;
  }
  free(b_matrix_path);

  if (convolve(a_matrix, b_matrix, &output_matrix)) {
    printf("convolve returned a non-zero integer\n");
    return -1;
  }

  char *output_matrix_path = get_output_matrix_path(task);
  if (write_matrix(output_matrix_path, output_matrix)) {
    printf("Error writing matrix to %s\n", output_matrix_path);
    return -1;
  }
  free(output_matrix_path);

  free(a_matrix->data);
  free(b_matrix->data);
  free(output_matrix->data);
  free(a_matrix);
  free(b_matrix);
  free(output_matrix);
  return 0;
}
