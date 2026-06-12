#!/usr/bin/env python3
"""
Verify neura-interpreter results against NumPy golden outputs.

For each model:
1. Parse memref.global constants from Neura .mlir (both hex and decimal formats)
2. Run neura-interpreter, extract input/output from simulated_memory
3. Compute golden output with NumPy
4. Compare
"""

import os, re, struct, subprocess, sys, json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))))  # model_tests/.. → python2neura/.. → Conversion/.. → test/.. → neura root
OUTPUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "output")  # output/ alongside model_tests/
INTERPRETER = os.path.join(WORKSPACE, "build/tools/neura-interpreter/neura-interpreter")

# ---------------------------------------------------------------------------
# MLIR parsing helpers
# ---------------------------------------------------------------------------

def parse_global_dense(text: str) -> np.ndarray:
    """Parse a memref.global constant value into numpy array.
    
    Supports two formats:
      hex:   dense<"0x000000003F800000...">
      decimal: dense<[0.0, 0.125, ...]> or dense<[[...],[...]]>
    """
    text = text.strip()
    
    # Try hex format first
    hex_m = re.search(r'dense<\s*"([0-9A-Fa-fx\s]+)"', text)
    if hex_m:
        hex_str = hex_m.group(1).strip()
        hex_str = hex_str.replace(' ', '').replace('0x', '').replace('0X', '')
        raw = bytes.fromhex(hex_str)
        return np.array(struct.unpack(f'<{len(raw)//4}f', raw), dtype=np.float32)
    
    # Decimal format: dense<[0.0, 1.0, ...]> or dense<[[...],[...]]>
    dec_m = re.search(r'dense<\s*(\[\[.*?\]\]|\s*\[.*?\])\s*>', text, re.DOTALL)
    if dec_m:
        arr_str = dec_m.group(1)
        # Extract all float-like numbers
        nums = re.findall(r'[+-]?\d+\.?\d*(?:[eE][+-]?\d+)?', arr_str)
        vals = [float(n) for n in nums]
        return np.array(vals, dtype=np.float32)
    
    return None


def parse_globals_from_neura(filepath: str) -> dict:
    """Parse all memref.global constants from a Neura .mlir file."""
    with open(filepath) as f:
        content = f.read()
    
    # Pattern: memref.global "private" constant @NAME : memref<SHAPE> = dense<...>
    pattern = r'memref\.global\s+"private"\s+constant\s+@(\w+)\s*:\s*memref<([^>]+)>\s*=\s*dense<(.+?)>\s*\{'
    globals_data = {}
    for m in re.finditer(pattern, content, re.DOTALL):
        name = m.group(1)
        shape_str = m.group(2)
        dense_content = m.group(3)
        dense_str = f'dense<{dense_content}>'
        
        # Clean shape: strip 'f32', 'x', and parse dims
        dims = [d.strip() for d in shape_str.split('x')]
        dims = [int(d) for d in dims if d.isdigit()]
        shape = tuple(dims)
        
        data = parse_global_dense(dense_str)
        if data is not None and shape and np.prod(shape) == data.size:
            data = data.reshape(shape)
        globals_data[name] = (shape, data)
        print(f"  Global '{name}': shape={data.shape}, range=[{data.min():.6f}, {data.max():.6f}]")
    
    return globals_data


def seed_input_data(shape: tuple) -> np.ndarray:
    """Reproduce the interpreter's hash-based random seed for %input0.
    
    The interpreter uses:
      hash = 0x9e3779b9
      for each coord: hash ^= coord + 0x9e3779b9 + (hash << 6) + (hash >> 2)
      float = (int32_t)(hash & 0xFFFFFFFF) / 2147483648.0f
    """
    def to_int32(x):
        # Simulate C++ int32_t truncation
        x = x & 0xFFFFFFFF
        if x >= 0x80000000:
            x -= 0x100000000
        return x
    
    arr = np.zeros(shape, dtype=np.float32)
    it = np.nditer(arr, flags=['multi_index'], op_flags=['writeonly'])
    for _ in it:
        idx = it.multi_index
        h = 0x9e3779b9
        for c in idx:
            # Simulate 32-bit unsigned arithmetic
            h = (h ^ (c + 0x9e3779b9 + ((h << 6) & 0xFFFFFFFF) + ((h >> 2) & 0xFFFFFFFF))) & 0xFFFFFFFF
        val = to_int32(h) / 2147483648.0
        arr[idx] = np.float32(val)
    return arr


def run_interpreter_get_memory(mlir_file: str) -> dict:
    """Run interpreter and parse all store_indexed values.

    Keys use memref-ID format: ``m<id>/[idx0][idx1]...``
    Returns a dict grouped by memref ID: {mid: {(i,j): value, ...}, ...}
    An extra key ``__last_mid`` records the memref ID of the last store
    (which is the output memref for the final kernel).
    """
    result = subprocess.run(
        [INTERPRETER, mlir_file, "--verbose"],
        capture_output=True, text=True
    )
    # Pattern: Store to m<id>/[idx0][idx1]...: value = X.XX
    pattern = r'Store to (m\d+)/((?:\[\d+\])+): value = ([-\d.e+]+)'
    memory = {}  # mid -> {(indices): value}
    last_mid = None
    for line in result.stdout.split('\n'):
        m = re.search(pattern, line)
        if m:
            mid = m.group(1)
            idx_str = m.group(2)
            indices = tuple(int(x) for x in re.findall(r'\[(\d+)\]', idx_str))
            val = float(m.group(3))
            memory.setdefault(mid, {})[indices] = val
            last_mid = mid
    if last_mid:
        memory['__last_mid'] = last_mid
    return memory


def memory_to_ndarray(memory: dict, shape: tuple) -> np.ndarray:
    """Find the output memref-ID group and convert to a dense numpy array.

    Uses ``__last_mid`` if available (the memref written by the last store
    in execution order), otherwise falls back to the last group matching shape.
    Returns a zero-filled array if none found.
    """
    arr = np.zeros(shape, dtype=np.float32)
    shape_len = len(shape)

    # Prefer the memref written by the very last store (final output).
    last_mid = memory.pop('__last_mid', None)
    if last_mid and last_mid in memory:
        coords = memory[last_mid]
        for idx, v in coords.items():
            if all(i < s for i, s in zip(idx, shape)):
                arr[idx] = v
        if np.count_nonzero(arr) > 0 or len(coords) == np.prod(shape):
            return arr

    # Fallback: iterate in reverse insertion order to pick the latest match.
    for mid, coords in reversed(list(memory.items())):
        if not coords:
            continue
        max_dims = [max(idx[d] for idx in coords.keys()) + 1 for d in range(shape_len)]
        if tuple(max_dims) == shape:
            for idx, v in coords.items():
                if all(i < s for i, s in zip(idx, shape)):
                    arr[idx] = v
            return arr
    for mid, coords in reversed(list(memory.items())):
        if not coords:
            continue
        sample = next(iter(coords.keys()))
        if len(sample) == shape_len:
            for idx, v in coords.items():
                if all(i < s for i, s in zip(idx, shape)):
                    arr[idx] = v
            break
    return arr


def get_input_shape_from_neura(filepath: str) -> tuple:
    """Get the first argument's memref shape from function signature."""
    with open(filepath) as f:
        content = f.read()
    m = re.search(r'func\.func @\w+\(%arg0:\s*memref<([^>]+)>', content)
    if m:
        dims = m.group(1).strip().split('x')
        return tuple(int(d.strip()) for d in dims if d.strip().isdigit())
    return None


def get_output_shape_from_neura(filepath: str) -> tuple:
    """Get return memref shape."""
    with open(filepath) as f:
        content = f.read()
    m = re.search(r'->\s*memref<([^>]+)>', content)
    if m:
        dims = m.group(1).strip().split('x')
        return tuple(int(d.strip()) for d in dims if d.strip().isdigit())
    return None


# ---------------------------------------------------------------------------
# Golden computation
# ---------------------------------------------------------------------------

def softmax(x, axis=-1):
    x = x.astype(np.float64)
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return (e / np.sum(e, axis=axis, keepdims=True)).astype(np.float32)


def layernorm(x, gamma, beta, eps=1e-5):
    x = x.astype(np.float64)
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + eps) * gamma + beta).astype(np.float32)


def gelu(x):
    x = x.astype(np.float64)
    c = np.sqrt(2.0 / np.pi)
    return (0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x**3)))).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-model verification
# ---------------------------------------------------------------------------

def compare(golden: np.ndarray, interp: np.ndarray, name: str, rtol=1e-5, atol=1e-6):
    """Compare golden and interpreter output, print results."""
    diff = np.abs(golden - interp)
    matching = np.isclose(golden, interp, rtol=rtol, atol=atol)
    match_count = np.count_nonzero(matching)
    total = golden.size
    
    print(f"  Matches: {match_count}/{total}")
    print(f"  Max absolute diff: {diff.max():.10f}")
    print(f"  Mean absolute diff: {diff.mean():.10f}")
    
    if match_count < total:
        mismatch_idx = np.where(~matching)
        n_show = min(5, len(mismatch_idx[0]))
        for i in range(n_show):
            idx = tuple(mismatch_idx[j][i] for j in range(len(mismatch_idx)))
            print(f"    [{idx}] golden={golden[idx]:.10f} interp={interp[idx]:.10f}")
        return False
    print("  ✓ ALL MATCH!")
    return True


def verify_simple_matmul():
    """input(4x8) @ W(8x16) -> 4x16"""
    name = "simple_matmul"
    print(f"\n{'='*60}")
    print(f"Verifying: {name}")
    print(f"{'='*60}")
    
    neura_file = os.path.join(OUTPUT_DIR, f"{name}_neura_clean.mlir")
    if not os.path.exists(neura_file):
        print("  SKIP")
        return False
    
    globals_data = parse_globals_from_neura(neura_file)
    if len(globals_data) < 1:
        print("  ERROR: no globals")
        return False
    
    weight_name = list(globals_data.keys())[0]
    weight = globals_data[weight_name][1]
    
    input_shape = get_input_shape_from_neura(neura_file) or (4, 8)
    output_shape = get_output_shape_from_neura(neura_file) or (4, 16)
    
    memory = run_interpreter_get_memory(neura_file)
    # Input is seeded with hash-based random values by the interpreter
    input_data = seed_input_data(input_shape)
    output_interp = memory_to_ndarray(memory, output_shape)
    
    golden = input_data @ weight
    
    print(f"  Input shape: {input_shape}, Weight shape: {weight.shape}, Output shape: {output_shape}")
    print(f"  Input[0]: {input_data[0][:4]}...")
    print(f"  Golden[0]: {golden[0][:4]}...")
    print(f"  Interp[0]: {output_interp[0][:4]}...")
    
    return compare(golden, output_interp, name)


def verify_residual_block():
    """relu(input(4x8) @ W(8x8) + bias(8)) + input -> 4x8"""
    name = "residual_block"
    print(f"\n{'='*60}")
    print(f"Verifying: {name}")
    print(f"{'='*60}")
    
    neura_file = os.path.join(OUTPUT_DIR, f"{name}_neura_clean.mlir")
    if not os.path.exists(neura_file):
        print("  SKIP")
        return False
    
    globals_data = parse_globals_from_neura(neura_file)
    if len(globals_data) < 2:
        print(f"  ERROR: need 2 globals, got {len(globals_data)}")
        return False
    
    # Sort by size: smaller is bias(8), larger is weight(8x8)
    sorted_globals = sorted(globals_data.items(), key=lambda x: np.prod(x[1][0]))
    bias = sorted_globals[0][1][1]
    weight = sorted_globals[1][1][1]
    
    input_shape = get_input_shape_from_neura(neura_file) or (4, 8)
    output_shape = get_output_shape_from_neura(neura_file) or (4, 8)
    
    memory = run_interpreter_get_memory(neura_file)
    input_data = seed_input_data(input_shape)
    output_interp = memory_to_ndarray(memory, output_shape)
    
    # Golden: relu(input @ W + bias) + input
    h = np.maximum(input_data @ weight + bias, 0.0)
    golden = input_data + h
    
    print(f"  Input shape: {input_shape}, Output shape: {output_shape}")
    print(f"  Input[0]: {input_data[0]}")
    print(f"  Golden[0]: {golden[0]}")
    print(f"  Interp[0]: {output_interp[0]}")
    
    return compare(golden, output_interp, name)


def verify_two_layer_mlp():
    """relu(input(2x8) @ W1(8x16)) @ W2(16x4) -> 2x4"""
    name = "two_layer_mlp"
    print(f"\n{'='*60}")
    print(f"Verifying: {name}")
    print(f"{'='*60}")
    
    neura_file = os.path.join(OUTPUT_DIR, f"{name}_neura_clean.mlir")
    if not os.path.exists(neura_file):
        print("  SKIP")
        return False
    
    globals_data = parse_globals_from_neura(neura_file)
    if len(globals_data) < 2:
        print(f"  ERROR: need 2 globals, got {len(globals_data)}")
        return False
    
    sorted_globals = sorted(globals_data.items(), key=lambda x: np.prod(x[1][0]))
    w2 = sorted_globals[0][1][1]  # 16x4 (64 elements)
    w1 = sorted_globals[1][1][1]  # 8x16 (128 elements)
    
    input_shape = get_input_shape_from_neura(neura_file) or (2, 8)
    output_shape = get_output_shape_from_neura(neura_file) or (2, 4)
    
    memory = run_interpreter_get_memory(neura_file)
    input_data = seed_input_data(input_shape)
    output_interp = memory_to_ndarray(memory, output_shape)
    
    golden = np.maximum(input_data @ w1, 0.0) @ w2
    
    print(f"  Input shape: {input_shape}, W1: {w1.shape}, W2: {w2.shape}")
    print(f"  Input: {input_data}")
    print(f"  Golden: {golden}")
    print(f"  Interp: {output_interp}")
    
    return compare(golden, output_interp, name)


def verify_conv2d_relu_pool():
    """maxpool(relu(input(1x9) @ kernel(9x16)), 2) -> 1x8"""
    name = "conv2d_relu_pool"
    print(f"\n{'='*60}")
    print(f"Verifying: {name}")
    print(f"{'='*60}")
    
    neura_file = os.path.join(OUTPUT_DIR, f"{name}_neura_clean.mlir")
    if not os.path.exists(neura_file):
        print("  SKIP")
        return False
    
    globals_data = parse_globals_from_neura(neura_file)
    if len(globals_data) < 1:
        print("  ERROR: no globals")
        return False
    
    kernel = list(globals_data.values())[0][1]
    
    input_shape = get_input_shape_from_neura(neura_file) or (1, 9)
    output_shape = get_output_shape_from_neura(neura_file) or (1, 8)
    
    memory = run_interpreter_get_memory(neura_file)
    input_data = seed_input_data(input_shape)
    output_interp = memory_to_ndarray(memory, output_shape)
    
    # Golden: maxpool(relu(input @ kernel), pool=2)
    conv_out = np.maximum(input_data @ kernel, 0.0)
    n, c = conv_out.shape
    golden = np.zeros((n, c // 2), dtype=np.float32)
    for i in range(n):
        for j in range(0, c, 2):
            golden[i, j // 2] = np.max(conv_out[i, j:j+2])
    
    print(f"  Input: {input_data}")
    print(f"  Conv+ReLU: {conv_out}")
    print(f"  Golden: {golden}")
    print(f"  Interp: {output_interp}")
    
    return compare(golden, output_interp, name)


def verify_transformer_attention():
    """input(4x8) -> QKV proj -> attention -> output -> 4x8
    
    Note: The Neura MLIR approximates softmax with ReLU.
    The golden must match what the MLIR actually computes.
    """
    name = "transformer_attention"
    print(f"\n{'='*60}")
    print(f"Verifying: {name}")
    print(f"{'='*60}")
    
    neura_file = os.path.join(OUTPUT_DIR, f"{name}_neura_clean.mlir")
    if not os.path.exists(neura_file):
        print("  SKIP")
        return False
    
    globals_data = parse_globals_from_neura(neura_file)
    if len(globals_data) < 3:
        print(f"  ERROR: need >=3 globals, got {len(globals_data)}")
        return False
    
    # Get the order from get_global ops
    with open(neura_file) as f:
        content = f.read()
    get_global_order = re.findall(r'memref\.get_global\s*@(\w+)', content)
    print(f"  get_global order: {get_global_order}")
    
    if len(get_global_order) < 3:
        print("  ERROR: need 3 get_global ops")
        return False
    
    wq = globals_data[get_global_order[0]][1]
    wk = globals_data[get_global_order[1]][1]
    wv = globals_data[get_global_order[2]][1]
    
    input_shape = get_input_shape_from_neura(neura_file) or (4, 8)
    output_shape = get_output_shape_from_neura(neura_file) or (4, 8)
    
    memory = run_interpreter_get_memory(neura_file)
    input_data = seed_input_data(input_shape)
    
    output_interp = memory_to_ndarray(memory, output_shape)
    
    Q = input_data @ wq
    K = input_data @ wk
    V = input_data @ wv
    
    # The MLIR uses ReLU on attention scores instead of true softmax.
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    attn_weights = np.maximum(scores, 0.0)  # ReLU (matches MLIR)
    golden = attn_weights @ V
    
    print(f"  Input: {input_data}")
    print(f"  Golden (ReLU-softmax): {golden}")
    print(f"  Interp: {output_interp}")
    
    return compare(golden, output_interp, name, rtol=1e-3, atol=1e-4)


def verify_gelu_layernorm():
    """LayerNorm(input(4x8), gamma(8), beta(8)) -> 4x8 (MLIR has no GELU kernel)"""
    name = "gelu_layernorm"
    print(f"\n{'='*60}")
    print(f"Verifying: {name}")
    print(f"{'='*60}")
    
    neura_file = os.path.join(OUTPUT_DIR, f"{name}_neura_clean.mlir")
    if not os.path.exists(neura_file):
        print("  SKIP")
        return False
    
    globals_data = parse_globals_from_neura(neura_file)
    if len(globals_data) < 2:
        print(f"  ERROR: need 2 globals, got {len(globals_data)}")
        return False
    
    with open(neura_file) as f:
        content = f.read()
    get_global_order = re.findall(r'memref\.get_global\s*@(\w+)', content)
    print(f"  get_global order: {get_global_order}")
    
    if len(get_global_order) < 2:
        print("  ERROR: need 2 get_global ops")
        return False
    
    gamma = globals_data[get_global_order[0]][1]
    beta = globals_data[get_global_order[1]][1]
    
    input_shape = get_input_shape_from_neura(neura_file) or (4, 8)
    output_shape = get_output_shape_from_neura(neura_file) or (4, 8)
    
    memory = run_interpreter_get_memory(neura_file)
    input_data = seed_input_data(input_shape)
    output_interp = memory_to_ndarray(memory, output_shape)
    
    # MLIR only computes LayerNorm (no GELU kernel exists in the IR)
    golden = layernorm(input_data, gamma, beta)
    
    print(f"  Input[0]: {input_data[0]}")
    print(f"  Golden (LayerNorm)[0]: {golden[0]}")
    print(f"  Interp[0]: {output_interp[0]}")
    
    return compare(golden, output_interp, name, rtol=1e-3, atol=1e-4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results = {}
    
    results['simple_matmul'] = verify_simple_matmul()
    results['residual_block'] = verify_residual_block()
    results['two_layer_mlp'] = verify_two_layer_mlp()
    results['conv2d_relu_pool'] = verify_conv2d_relu_pool()
    results['transformer_attention'] = verify_transformer_attention()
    results['gelu_layernorm'] = verify_gelu_layernorm()
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {name:30s} {status}")
    print(f"\n  {passed}/{total} models passed")


if __name__ == "__main__":
    main()
