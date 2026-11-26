#define N 32

// Every CGRA runs this kernel on its own slice of the input; the partial sums
// and the partial sums of squares are each summed across CGRAs.
void kernel(int input[], int *sum, int *sum_of_squares) {
  int partial_sum = 0;
  int partial_sum_of_squares = 0;
  for (int i = 0; i < N; ++i) {
    partial_sum += input[i];
    partial_sum_of_squares += input[i] * input[i];
  }
  *sum = partial_sum;
  *sum_of_squares = partial_sum_of_squares;
}
