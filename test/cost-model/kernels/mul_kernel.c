// Constructed case: FU-class saturation (per-class ResMII), large II.
// Each iteration issues 10 INDEPENDENT integer multiplies into 10 SEPARATE
// accumulators, so every loop-carried recurrence is a length-2 add cycle
// (RecMII = 2) and does NOT mask the FU bound. On arch/scarce_mul.yaml, where
// integer 'mul' exists on a single tile, ResMII(mul) = ceil(10 / 1) = 10, so
// the mapper must serialise all 10 muls onto one tile => oracle II >= 10.
// The crude ceil(#ops/#tiles) baseline predicts ~2; the per-FU-class ResMII
// predicts 10. This is the flagship demonstration that per-class ResMII is
// necessary and predicts large II from FU scarcity.
long mul_kernel(long *a, int n) {
  long s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0;
  long s5 = 0, s6 = 0, s7 = 0, s8 = 0, s9 = 0;
  for (int i = 0; i < n; i++) {
    s0 = s0 + a[i + 0] * a[i + 0];
    s1 = s1 + a[i + 1] * a[i + 1];
    s2 = s2 + a[i + 2] * a[i + 2];
    s3 = s3 + a[i + 3] * a[i + 3];
    s4 = s4 + a[i + 4] * a[i + 4];
    s5 = s5 + a[i + 5] * a[i + 5];
    s6 = s6 + a[i + 6] * a[i + 6];
    s7 = s7 + a[i + 7] * a[i + 7];
    s8 = s8 + a[i + 8] * a[i + 8];
    s9 = s9 + a[i + 9] * a[i + 9];
  }
  return s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9;
}
