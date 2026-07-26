// Constructed case: memory-FU saturation (MemMII / mem-class ResMII).
// Each iteration performs 16 INDEPENDENT loads into 16 SEPARATE accumulators,
// so every loop-carried recurrence is a length-2 add cycle (RecMII = 2) and
// does NOT mask the memory bound. The default architecture provides memory FUs
// on only 7 of 16 tiles, so MemMII = ceil(16 / 7) = 3 and the mapper must
// serialise the 16 loads onto the 7 memory tiles => oracle II >= 3. Shows that
// the per-FU-class memory count (7 mem tiles, not 16) binds separately from the
// crude ceil(#ops/#tiles) baseline (which would predict ~2).
long mem_kernel(long *a, int n) {
  long s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
  long s8 = 0, s9 = 0, s10 = 0, s11 = 0, s12 = 0, s13 = 0, s14 = 0, s15 = 0;
  for (int i = 0; i < n; i++) {
    s0 = s0 + a[i + 0];
    s1 = s1 + a[i + 1];
    s2 = s2 + a[i + 2];
    s3 = s3 + a[i + 3];
    s4 = s4 + a[i + 4];
    s5 = s5 + a[i + 5];
    s6 = s6 + a[i + 6];
    s7 = s7 + a[i + 7];
    s8 = s8 + a[i + 8];
    s9 = s9 + a[i + 9];
    s10 = s10 + a[i + 10];
    s11 = s11 + a[i + 11];
    s12 = s12 + a[i + 12];
    s13 = s13 + a[i + 13];
    s14 = s14 + a[i + 14];
    s15 = s15 + a[i + 15];
  }
  return s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9 + s10 + s11 + s12 +
         s13 + s14 + s15;
}
