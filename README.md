# GPU Runtime Toolkit

GPU runtime diagnostics, compatibility testing, and inference optimization tools for local AI systems.

A standalone Python CLI that validates actual CUDA, PyTorch SDPA, Triton and SageAttention execution. It uses the selected existing Python environment without installing, updating or removing its packages. ComfyUI is not required.

---

- GPU inventory, compute capability, CUDA build and driver information
- Available GPU memory and reported native BF16 capability
- Actual FP32 matrix multiplication, FP16 SDPA and Triton kernel checks
- Architecture-aware SageAttention 2 INT8 QK / FP16 and FP8 PV probes
- Output shape, finite values and relative L2 error against a mathematical FP32 attention reference
- Separate subprocess and timeout for each GPU probe
- Multi-GPU selection and machine-readable JSON reports

### Requirements

Python 3.10+ and an existing NVIDIA CUDA-enabled PyTorch environment.
Triton and SageAttention are optional; their versions must be compatible with
the selected Python, GPU, PyTorch, driver and CUDA build. No automatic package
installation is performed.

Windows and Linux, x86-64 and ARM64 are target environments; actual support
depends on installed packages. This initial release candidate requires hardware
validation. GB10 unified memory is reported as CUDA-visible memory, not dedicated VRAM.

### Quick start

Clone the repository or extract the source archive. Run from its directory.

### ComfyUI

Open PowerShell in the folder containing `runtime.py`. When prompted, paste the
path to the ComfyUI portable folder that contains `python_embeded`.

```powershell
$comfyRoot = (Read-Host 'ComfyUI portable folder').Trim().Trim('"')
& (Join-Path $comfyRoot 'python_embeded\python.exe') .\runtime.py
```

### Windows / Linux — existing Python environment

```shell
python runtime.py
python runtime.py --device 1
python runtime.py --all-devices
python runtime.py --json
python runtime.py --output runtime-report.json
python runtime.py --all-devices --timeout 300 --output runtime-report.json
```

Use the Python interpreter that already has your GPU libraries installed.
`--device` uses PyTorch's logical CUDA index, respecting `CUDA_VISIBLE_DEVICES`.
`--timeout` is per subprocess, including library import and kernel compilation.
The first Triton compile may take longer; the installed libraries may create
normal compilation caches.

### GPU Compatibility

The default routes below are conservative, based on SageAttention 2 upstream
dispatch and selected secondary probes. They are test candidates, not confirmed
compatibility claims.

| Architecture | Compute Capability | Attention Backends |
|---|---|---|
| Turing | SM75 | None; test CUDA/SDPA/Triton |
| Ampere (A100) | SM80 | Auto, FP16 CUDA, FP16 Triton |
| Ampere (RTX 30) | SM86 | Auto, FP16 CUDA, FP16 Triton |
| Ada (RTX 40) | SM89 | Auto, FP16 CUDA/Triton, FP8 CUDA |
| Hopper | SM90 | Auto, FP16 Triton, Hopper FP8 CUDA |
| Blackwell (B100/B200/B300 class) | SM100/SM103 | No assumed SageAttention 2 route |
| Blackwell (RTX 50 / RTX PRO) | SM120 | Auto, FP8 CUDA |
| Grace Blackwell (GB10) | SM121 | Auto, FP8 CUDA |

FP8 CUDA uses `fp32+fp16` accumulation and, on SM120/SM121, `per_warp`
QK quantization. The Hopper-specific path uses `fp32+fp32` accumulation.
These are SageAttention internal QK/PV precision modes, not FP8 model-weight tests.

To explicitly attempt APIs outside the conservative architecture routes:

```shell
python runtime.py --probe-unlisted --timeout 300
```

Experimental probing can fail, time out, or trigger a driver reset on an
incompatible build. Process isolation protects the parent against ordinary
worker crashes, but cannot prevent system-wide driver failures. Save work first.
SageAttention 3 FP4, FlashAttention, model-weight FP8/NVFP4 and performance
benchmarks are outside this version's scope.

### Results

| Status | Meaning |
|---|---|
| PASS | Probe completed and its synthetic correctness check passed |
| SKIP | Optional dependency/API missing, route not selected, or prerequisite failed |
| FAIL | A probe or import failed, produced an invalid result, or crashed |
| TIMEOUT | Probe exceeded its time limit |

Exit codes: `0` = selected probes completed without failures (optional skips
allowed); `1` = at least one probe failed or timed out; `2` = inventory or
device selection unavailable.

The relative L2 thresholds (0.02 for SDPA, 0.25 for SageAttention) are basic
synthetic smoke-test limits, not model-quality criteria. A passing probe does
not establish the fastest backend or guarantee real model compatibility.
No optimization settings are automatically applied.

Package version metadata may be unavailable for manually installed modules.
The actual import and execution tests determine whether the selected API runs.
`elapsed_seconds` includes process startup and compilation; it is **not** a
kernel benchmark. JSON reports may contain system and local path details in
diagnostic logs; review them before sharing publicly.

### References

- [PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [Triton](https://triton-lang.org/)
- [SageAttention upstream dispatch](https://github.com/thu-ml/SageAttention/blob/main/sageattention/core.py)

### License

MIT. The dependencies retain their respective licenses.
