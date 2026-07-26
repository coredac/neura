// Constructed case: long loop-carried recurrence (large RecMII).
// Each iteration threads the accumulator through a long serial chain of
// data-dependent add/sub/mul operations (interleaving array reads a[i]/b[i] so
// the chain cannot be folded). The loop-carried dependence distance is 1, so
// RecMII = latency sum of the chain (~a dozen cycles). This exercises the
// model's prediction of LARGE II from a deep recurrence. Only add/sub/mul are
// used (the LLVM->Neura lowering supports these directly).
long chain_kernel(long *a, long *b, int n) {
  long acc = 1;
  for (int i = 0; i < n; i++) {
    long x = acc + a[i];
    x = x - b[i];
    x = x + a[i];
    x = x - b[i];
    x = x + a[i];
    x = x - b[i];
    x = x + a[i];
    x = x - b[i];
    x = x + a[i];
    x = x - b[i];
    acc = x;
  }
  return acc;
}
