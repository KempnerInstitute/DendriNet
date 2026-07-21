"""Triton environment smoke test: compile + run a trivial kernel on this GPU.

Goal: confirm Triton is installed, can JIT-compile a kernel, and produces correct
results on this A100 -- BEFORE we invest in the real fused-gather kernel. If this
fails, the failure is an environment/toolchain issue, not our algorithm.

Run:  python triton_smoketest.py
"""
import torch

print("=== environment ===")
print(f"torch      : {torch.__version__}")
print(f"cuda avail : {torch.cuda.is_available()}")
assert torch.cuda.is_available(), "no CUDA device visible"
dev = "cuda"
print(f"gpu        : {torch.cuda.get_device_name(0)}")
cc = torch.cuda.get_device_capability(0)
print(f"compute cap: {cc[0]}.{cc[1]}")

try:
    import triton
    import triton.language as tl
    print(f"triton     : {triton.__version__}")
except Exception as e:  # noqa: BLE001
    print(f"triton import FAILED: {e}")
    raise SystemExit(1)


# ---- the simplest possible kernel: elementwise vector add --------------------
@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)                 # which block am I?
    offs = pid * BLOCK + tl.arange(0, BLOCK)    # the indices this block handles
    mask = offs < n                             # guard the tail
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def triton_add(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)   # noqa: E731
    _add_kernel[grid](x, y, out, n, BLOCK=1024)
    return out


print("\n=== compile + run ===")
torch.manual_seed(0)
x = torch.randn(1_000_003, device=dev)          # deliberately not a multiple of BLOCK
y = torch.randn_like(x)

out = triton_add(x, y)                            # first call JIT-compiles the kernel
ref = x + y
max_err = (out - ref).abs().max().item()
ok = torch.allclose(out, ref, rtol=1e-5, atol=1e-6)
print(f"kernel compiled and ran: OK")
print(f"correctness vs torch   : {'OK' if ok else 'FAIL'}  (max_err={max_err:.2e})")

# ---- tiny timing sanity check (not a benchmark, just 'does it run fast') -----
for _ in range(5): triton_add(x, y)
torch.cuda.synchronize()
e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
e0.record()
for _ in range(50): triton_add(x, y)
e1.record(); torch.cuda.synchronize()
print(f"~{e0.elapsed_time(e1)/50*1e3:.1f} us/call for {x.numel()/1e6:.1f}M elements")

print("\nVERDICT:", "Triton works on this GPU -- clear to write the real kernel."
      if ok else "Triton ran but gave wrong results -- STOP, investigate.")
