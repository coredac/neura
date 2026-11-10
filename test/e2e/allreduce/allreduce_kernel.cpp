extern "C" int cgra_allreduce_add(int partial);

#define NTAPS 32

// Every CGRA runs this kernel on its own slice of the taps; the partial dot
// products are summed across CGRAs.
int kernel(int input[], int coefficient[]) {
  int partial = 0;
  for (int i = 0; i < NTAPS; ++i) {
    partial += input[i] * coefficient[i];
  }
  return cgra_allreduce_add(partial);
}
