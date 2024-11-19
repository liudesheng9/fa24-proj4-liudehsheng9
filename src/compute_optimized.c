#include <omp.h>
#include <x86intrin.h>

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


  //create and fill mat_flip, which is a flipped version of b
  matrix_t* mat_flip = malloc(sizeof(matrix_t));
  mat_flip->rows = b_matrix->rows;
  mat_flip->cols = b_matrix->cols;
  int mat_flip_size = b_matrix->rows * b_matrix->cols;
  mat_flip->data = malloc(sizeof(int32_t) * mat_flip_size);

  #pragma omp parallel for
  for(int i = 0; i < mat_flip_size; i++) {
    mat_flip->data[i] = b_matrix->data[mat_flip_size - 1 - i];
  }
  
  //set var
  uint32_t col_a = a_matrix -> cols;
  uint32_t col_b = b_matrix -> cols;
  uint32_t col_flip = mat_flip -> cols;
  int32_t* mat_data_flip = mat_flip->data;
  int32_t* mat_data_a = a_matrix->data;

  //convolve
  #pragma omp parallel for collapse(2)
  for (int r = 0; r < num_output_rows; r ++) {
    for (int c = 0; c < num_output_cols; c ++) {
        int thread_sum = 0;
        for (int i = 0; i < (mat_flip->rows); i++) {
          __m256i thread_sum_vec = _mm256_set1_epi32(0);
          /*
          for (int j = c; j < (col_flip-col_flip%64) + c; j+= 64) {
              __m256i flip_vec8 = _mm256_loadu_si256( (__m256i *) (mat_data_flip + (j-c) + (col_b*i)));
              __m256i a_vec8 = _mm256_loadu_si256( (__m256i *) (mat_data_a + j + (col_a*(i+r))));
              __m256i mul_vec8 = _mm256_mullo_epi32(flip_vec8, a_vec8);
              thread_sum_vec = _mm256_add_epi32(thread_sum_vec, mul_vec8);

              flip_vec8 = _mm256_loadu_si256( (__m256i *) (mat_data_flip + 8 + (j-c) + (col_b*i)));
              a_vec8 = _mm256_loadu_si256( (__m256i *) (mat_data_a + 8 + j + (col_a*(i+r))));
              mul_vec8 = _mm256_mullo_epi32(flip_vec8, a_vec8);
              thread_sum_vec = _mm256_add_epi32(thread_sum_vec, mul_vec8);
              
              flip_vec8 = _mm256_loadu_si256( (__m256i *) (mat_data_flip + 16 + (j-c) + (col_b*i)));
              a_vec8 = _mm256_loadu_si256( (__m256i *) (mat_data_a + 16 + j + (col_a*(i+r))));
              mul_vec8 = _mm256_mullo_epi32(flip_vec8, a_vec8);
              thread_sum_vec = _mm256_add_epi32(thread_sum_vec, mul_vec8);

              flip_vec8 = _mm256_loadu_si256( (__m256i *) (mat_data_flip + 24 + (j-c) + (col_b*i)));
              a_vec8 = _mm256_loadu_si256( (__m256i *) (mat_data_a + 24 + j + (col_a*(i+r))));
              mul_vec8 = _mm256_mullo_epi32(flip_vec8, a_vec8);
              thread_sum_vec = _mm256_add_epi32(thread_sum_vec, mul_vec8);

              flip_vec8 = _mm256_loadu_si256( (__m256i *) (mat_data_flip + 32 + (j-c) + (col_b*i)));
              a_vec8 = _mm256_loadu_si256( (__m256i *) (mat_data_a + 32 + j + (col_a*(i+r))));
              mul_vec8 = _mm256_mullo_epi32(flip_vec8, a_vec8);
              thread_sum_vec = _mm256_add_epi32(thread_sum_vec, mul_vec8);
              
              flip_vec8 = _mm256_loadu_si256( (__m256i *) (mat_data_flip + 40 + (j-c) + (col_b*i)));
              a_vec8 = _mm256_loadu_si256( (__m256i *) (mat_data_a + 40 + j + (col_a*(i+r))));
              mul_vec8 = _mm256_mullo_epi32(flip_vec8, a_vec8);
              thread_sum_vec = _mm256_add_epi32(thread_sum_vec, mul_vec8);

              flip_vec8 = _mm256_loadu_si256( (__m256i *) (mat_data_flip + 48 + (j-c) + (col_b*i)));
              a_vec8 = _mm256_loadu_si256( (__m256i *) (mat_data_a + 48 + j + (col_a*(i+r))));
              mul_vec8 = _mm256_mullo_epi32(flip_vec8, a_vec8);
              thread_sum_vec = _mm256_add_epi32(thread_sum_vec, mul_vec8);

              flip_vec8 = _mm256_loadu_si256( (__m256i *) (mat_data_flip + 56 + (j-c) + (col_b*i)));
              a_vec8 = _mm256_loadu_si256( (__m256i *) (mat_data_a + 56 + j + (col_a*(i+r))));
              mul_vec8 = _mm256_mullo_epi32(flip_vec8, a_vec8);
              thread_sum_vec = _mm256_add_epi32(thread_sum_vec, mul_vec8);
          }
          int j = col_flip-col_flip%64 + c;
          for (int j = c; j < (col_flip-col_flip%32)+c; j+= 32) {
              __m256i flip_vec4 = _mm256_loadu_si256( (__m256i *) (mat_data_flip + (j-c) + (col_b*i)));
              __m256i a_vec4 = _mm256_loadu_si256( (__m256i *) (mat_data_a + j + (col_a*(i+r))));
              __m256i mul_vec4 = _mm256_mullo_epi32(flip_vec4, a_vec4);
              thread_sum_vec = _mm256_add_epi32(thread_sum_vec, mul_vec4);

              flip_vec4 = _mm256_loadu_si256( (__m256i *) (mat_data_flip + 8 + (j-c) + (col_b*i)));
              a_vec4 = _mm256_loadu_si256( (__m256i *) (mat_data_a + 8 + j + (col_a*(i+r))));
              mul_vec4 = _mm256_mullo_epi32(flip_vec4, a_vec4);
              thread_sum_vec = _mm256_add_epi32(thread_sum_vec, mul_vec4);
              
              flip_vec4 = _mm256_loadu_si256( (__m256i *) (mat_data_flip + 16 + (j-c) + (col_b*i)));
              a_vec4 = _mm256_loadu_si256( (__m256i *) (mat_data_a + 16 + j + (col_a*(i+r))));
              mul_vec4 = _mm256_mullo_epi32(flip_vec4, a_vec4);
              thread_sum_vec = _mm256_add_epi32(thread_sum_vec, mul_vec4);

              flip_vec4 = _mm256_loadu_si256( (__m256i *) (mat_data_flip + 24 + (j-c) + (col_b*i)));
              a_vec4 = _mm256_loadu_si256( (__m256i *) (mat_data_a + 24 + j + (col_a*(i+r))));
              mul_vec4 = _mm256_mullo_epi32(flip_vec4, a_vec4);
              thread_sum_vec = _mm256_add_epi32(thread_sum_vec, mul_vec4);
          }
          j = col_flip-col_flip%32 + c;
          for (; j < (col_flip-col_flip%16)+c; j+= 16) {
            __m256i flip_vec2 = _mm256_loadu_si256( (__m256i *) (mat_data_flip + (j-c) + (col_b*i)));
            __m256i a_vec2 = _mm256_loadu_si256( (__m256i *) (mat_data_a + j + (col_a*(i+r))));
            __m256i mul_vec2 = _mm256_mullo_epi32(flip_vec2, a_vec2);
            thread_sum_vec = _mm256_add_epi32(thread_sum_vec, mul_vec2);

            flip_vec2 = _mm256_loadu_si256( (__m256i *) (mat_data_flip + 8 + (j-c) + (col_b*i)));
            a_vec2 = _mm256_loadu_si256( (__m256i *) (mat_data_a + 8 + j + (col_a*(i+r))));
            mul_vec2 = _mm256_mullo_epi32(flip_vec2, a_vec2);
            thread_sum_vec = _mm256_add_epi32(thread_sum_vec, mul_vec2);
          }
          j = col_flip-col_flip%16 + c;
          */
          for (int j=c; j < (col_flip-col_flip%8)+c; j+= 8) {
            __m256i flip_vec1 = _mm256_loadu_si256((__m256i *) (mat_data_flip + (j-c) + (col_b*i)));
            __m256i a_vec1 = _mm256_loadu_si256((__m256i *) (mat_data_a + j + (col_a*(i+r))));
            __m256i mul_vec1 = _mm256_mullo_epi32(flip_vec1, a_vec1);
            thread_sum_vec = _mm256_add_epi32(thread_sum_vec, mul_vec1);
          }
          int j = col_flip-col_flip%8 + c;
          for(; j < (col_flip)+c; j++) {
            thread_sum += (a_matrix->data[j + (col_a*(i+r))])*(mat_flip->data[j-c + (col_b*i)]);
          }
          int int_thread_sum[8];
          _mm256_storeu_si256((__m256i *) int_thread_sum, thread_sum_vec);
          thread_sum += int_thread_sum[0] + int_thread_sum[1] + int_thread_sum[2] + int_thread_sum[3] + int_thread_sum[4] + int_thread_sum[5] + int_thread_sum[6] + int_thread_sum[7];
          
        }
        (*output_matrix)->data[c + r*num_output_cols] = thread_sum;
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
