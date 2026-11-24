"""A minimal real Triton compile-and-execute probe."""
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(x, y, z, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < n
    tl.store(z + i, tl.load(x + i, mask=mask, other=0)
             + tl.load(y + i, mask=mask, other=0), mask=mask)

def probe(torch, device):
    n = 4099
    x = torch.arange(n, device=device, dtype=torch.float32)
    y = torch.ones_like(x)
    z = torch.empty_like(x)
    vector_add_kernel[(triton.cdiv(n, 256),)](x, y, z, n, BLOCK=256)
    torch.cuda.synchronize(device)
    if not torch.equal(z, x + y):
        raise RuntimeError("Triton result mismatch")
    return "Triton compilation and vector addition verified"
