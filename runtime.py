"""SOLRICKS AI Runtime — isolated GPU compatibility diagnostics."""
import argparse
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

VERSION = "0.1.0"
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TESTS = {
    "cuda": "CUDA FP32 matrix multiplication",
    "sdpa": "PyTorch SDPA (FP16)",
    "triton": "Triton vector addition",
    "sage-auto": "SageAttention Auto",
    "sage-triton": "SageAttention INT8 QK / FP16 PV Triton",
    "sage-fp16": "SageAttention INT8 QK / FP16 PV CUDA",
    "sage-fp8": "SageAttention INT8 QK / FP8 PV CUDA",
    "sage-sm90": "SageAttention INT8 QK / FP8 PV CUDA SM90",
}
ARCHITECTURES = {
    (5, 0): "Maxwell", (5, 2): "Maxwell", (5, 3): "Maxwell",
    (6, 0): "Pascal", (6, 1): "Pascal", (6, 2): "Pascal",
    (7, 0): "Volta", (7, 2): "Volta", (7, 5): "Turing",
    (8, 0): "Ampere", (8, 6): "Ampere", (8, 7): "Ampere",
    (8, 9): "Ada Lovelace", (9, 0): "Hopper",
    (10, 0): "Blackwell", (10, 3): "Blackwell",
    (12, 0): "Blackwell", (12, 1): "Blackwell",
}
# Conservative routing based on upstream SageAttention 2 auto-dispatch.
# Other architectures can be explicitly probed; never infer PASS from SM alone.
SAGE_AUTO = {(8, 0), (8, 6), (8, 9), (9, 0), (12, 0), (12, 1)}
SAGE_ROUTES = {
    "sage-auto": SAGE_AUTO,
    "sage-triton": {(8, 0), (8, 6), (8, 9), (9, 0)},
    "sage-fp16": {(8, 0), (8, 6), (8, 9)},
    "sage-fp8": {(8, 9), (12, 0), (12, 1)},
    "sage-sm90": {(9, 0)},
}

def sage_spec(test, sm):
    if test == "sage-auto":
        return "sageattn", {}
    if test == "sage-triton":
        return "sageattn_qk_int8_pv_fp16_triton", {}
    if test == "sage-fp16":
        return "sageattn_qk_int8_pv_fp16_cuda", {"pv_accum_dtype": "fp32"}
    if test == "sage-fp8":
        opts = {"pv_accum_dtype": "fp32+fp16"}
        if sm in {(10, 0), (10, 3), (12, 0), (12, 1)}:
            opts["qk_quant_gran"] = "per_warp"
        return "sageattn_qk_int8_pv_fp8_cuda", opts
    if test == "sage-sm90":
        return "sageattn_qk_int8_pv_fp8_cuda_sm90", {"pv_accum_dtype": "fp32+fp32"}
    raise ValueError(test)

def result(status, detail, **extra):
    return {"status": status, "detail": str(detail), **extra}

def installed_version(*names):
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None

def inventory():
    """Executed in a worker; CUDA initialization may fail in native code."""
    import torch
    data = {
        "torch": str(torch.__version__),
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "devices": [],
    }
    if data["cuda_available"]:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            sm = tuple(torch.cuda.get_device_capability(i))
            with torch.cuda.device(i):
                free, total = torch.cuda.mem_get_info(i)
                try:
                    bf16 = bool(torch.cuda.is_bf16_supported(including_emulation=False))
                except TypeError:
                    bf16 = None  # older torch cannot explicitly exclude emulation
            data["devices"].append({
                "index": i, "name": props.name, "sm": list(sm),
                "architecture": ARCHITECTURES.get(sm, "Unknown"),
                "device_memory_gib": round(props.total_memory / 2**30, 2),
                "cuda_free_gib": round(free / 2**30, 2),
                "cuda_total_gib": round(total / 2**30, 2),
                "native_bf16_reported": bf16,
            })
    return data

def worker(task, index, experimental):
    import importlib
    if task == "inventory":
        try:
            return result("PASS", "PyTorch inventory collected", data=inventory())
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                return result("SKIP", "PyTorch is not installed in this interpreter")
            raise
    import torch
    torch.cuda.set_device(index)
    device = torch.device("cuda", index)
    sm = tuple(torch.cuda.get_device_capability(index))
    with torch.inference_mode():
        if task == "cuda":
            a = torch.ones((256, 256), device=device, dtype=torch.float32)
            out = a @ a
            torch.cuda.synchronize(device)
            if not torch.equal(out, torch.full_like(out, 256)):
                raise RuntimeError("Incorrect matrix multiplication")
            return result("PASS", "FP32 GPU computation verified")

        if task == "triton":
            try:
                from triton_probe import probe
            except ModuleNotFoundError as exc:
                if exc.name == "triton":
                    return result("SKIP", "Triton is not installed")
                raise
            return result("PASS", probe(torch, device))

        function, options = None, {}
        if task.startswith("sage-"):
            if sm not in SAGE_ROUTES[task] and not experimental:
                return result("SKIP", f"SM{sm[0]}{sm[1]} is outside the default test route")
            try:
                sage = importlib.import_module("sageattention")
            except ModuleNotFoundError as exc:
                if exc.name == "sageattention":
                    return result("SKIP", "SageAttention is not installed")
                raise
            api, options = sage_spec(task, sm)
            function = getattr(sage, api, None)
            if not callable(function):
                return result("SKIP", f"Installed build does not expose {api}")

        # Reproducible synthetic inputs. Head dimension 128, non-causal.
        gen = torch.Generator(device=device)
        gen.manual_seed(42)
        q, k, v = [torch.randn((1, 8, 256, 128), device=device,
                                dtype=torch.float16, generator=gen)
                   for _ in range(3)]
        # Explicit FP32 mathematical reference, avoiding SDPA backend dispatch.
        # Disable TF32 for the reference in this isolated process.
        torch.backends.cuda.matmul.allow_tf32 = False
        ref = ((q.float() @ k.float().transpose(-2, -1))
               / math.sqrt(q.shape[-1])).softmax(dim=-1) @ v.float()
        if task == "sdpa":
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=False)
            threshold = 0.02
        else:
            out = function(q, k, v, tensor_layout="HND",
                           is_causal=False, **options)
            threshold = 0.25
        torch.cuda.synchronize(device)
        if not isinstance(out, torch.Tensor) or out.shape != q.shape:
            raise RuntimeError("Unexpected output type or shape")
        if out.device != q.device:
            raise RuntimeError("Output is not on the selected CUDA device")
        if not torch.isfinite(out).all().item():
            raise RuntimeError("Non-finite output")
        error = float(((out.float() - ref).norm()
                       / ref.norm().clamp_min(1e-8)).item())
        if not math.isfinite(error) or error > threshold:
            return result("FAIL", "Numerical sanity check failed",
                          relative_l2_error=error if math.isfinite(error) else None,
                          threshold=threshold)
        return result("PASS", f"GPU execution verified; relative L2 error={error:.6f}",
                      relative_l2_error=error, threshold=threshold,
                      shape=list(out.shape))

def run_worker(task, index=0, timeout=180, experimental=False):
    """One interpreter per probe. No native imports in the parent process."""
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="solricks-runtime-") as tmp:
        path = Path(tmp) / "result.json"
        cmd = [sys.executable, str(ROOT / "runtime.py"), "--_worker", task,
               "--_result", str(path), "--device", str(index)]
        if experimental:
            cmd.append("--probe-unlisted")
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=timeout)
            if proc.returncode != 0:
                data = result("FAIL", f"Worker exited with code {proc.returncode}",
                              log_tail=(proc.stderr + "\n" + proc.stdout)[-3000:])
            elif not path.exists():
                data = result("FAIL", "Worker produced no result",
                              log_tail=(proc.stderr + "\n" + proc.stdout)[-3000:])
            else:
                data = json.loads(path.read_text(encoding="utf-8"))
                if proc.stderr.strip():
                    data["log_tail"] = proc.stderr[-3000:]
        except subprocess.TimeoutExpired:
            data = result("TIMEOUT", f"Worker exceeded {timeout} seconds")
        except Exception as exc:
            data = result("FAIL", f"{type(exc).__name__}: {exc}")
    data["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return data

def driver_info():
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=10, check=True)
        return p.stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return []

def diagnostics(device, all_devices=False, timeout=180, experimental=False):
    report = {
        "version": VERSION,
        "system": {"os": platform.platform(), "python": platform.python_version(),
                   "architecture": platform.machine(), "nvidia_smi": driver_info()},
        "packages": {
            "torch": installed_version("torch"),
            "triton": installed_version("triton", "triton-windows"),
            "sageattention": installed_version("sageattention", "sageattention-windows"),
        },
        "inventory": run_worker("inventory", timeout=timeout),
        "gpus": [],
        "recommendations": [],
        "exit_code": 0,
    }
    inv = report["inventory"]
    if inv["status"] != "PASS":
        report["exit_code"] = 2
        report["recommendations"].append("Use an existing CUDA-enabled PyTorch interpreter.")
        return report
    devices = inv["data"]["devices"]
    if not devices:
        report["exit_code"] = 2
        report["recommendations"].append("No CUDA device available to PyTorch.")
        return report
    selected = devices if all_devices else [g for g in devices if g["index"] == device]
    if not selected:
        report["exit_code"] = 2
        report["recommendations"].append(f"Invalid CUDA device index: {device}")
        return report
    for gpu in selected:
        sm = tuple(gpu["sm"])
        tests = []
        for task, name in TESTS.items():
            if task in SAGE_ROUTES and sm not in SAGE_ROUTES[task] and not experimental:
                entry = result("SKIP", f"Not in default SageAttention route for SM{sm[0]}{sm[1]}")
            elif tests and tests[0]["status"] != "PASS":
                entry = result("SKIP", "Basic CUDA test did not pass")
            else:
                entry = run_worker(task, gpu["index"], timeout, experimental)
            tests.append({"id": task, "name": name, **entry})
        passed = [t["name"] for t in tests if t["status"] == "PASS"
                  and (t["id"].startswith("sage-") or t["id"] == "sdpa")]
        report["gpus"].append({**gpu, "tests": tests,
                               "validated_attention_candidates": passed})
        if any(t["status"] in {"FAIL", "TIMEOUT"} for t in tests):
            report["exit_code"] = 1
        if gpu["cuda_free_gib"] < 4:
            report["recommendations"].append(
                f"GPU {gpu['index']}: less than 4 GiB free at inventory time.")
        if sm in {(10, 0), (10, 3)}:
            report["recommendations"].append(
                "Data-center Blackwell: default SageAttention 2 routes are not assumed. "
                "Use --probe-unlisted with a compatible build if desired.")
    report["recommendations"].append(
        "PASS validates only this synthetic workload, not speed, model quality or full compatibility.")
    return report

def print_report(report):
    print(f"SOLRICKS AI Runtime v{VERSION}")
    print("Build. Optimize. Deploy.")
    print("=" * 72)
    for key, value in report["system"].items():
        print(f"{key}: {value}")
    print("\nPACKAGES (distribution metadata)")
    for key, value in report["packages"].items():
        print(f"{key}: {value or 'Not detected'}")
    inv = report["inventory"]
    print(f"\nINVENTORY [{inv['status']}] {inv['detail']}")
    if "data" in inv:
        print(f"PyTorch: {inv['data']['torch']} | CUDA build: {inv['data']['torch_cuda_build']}")
    for gpu in report["gpus"]:
        sm = gpu["sm"]
        print(f"\nGPU {gpu['index']}: {gpu['name']} | {gpu['architecture']} SM{sm[0]}{sm[1]}")
        print(f"CUDA memory: {gpu['cuda_free_gib']} / {gpu['cuda_total_gib']} GiB free")
        print(f"Native BF16 reported: {gpu['native_bf16_reported']}")
        print("-" * 72)
        for t in gpu["tests"]:
            print(f"[{t['status']:<7}] {t['name']}")
            print(f"          {t['detail']}")
            if t["status"] in {"FAIL", "TIMEOUT"} and t.get("log_tail"):
                print(t["log_tail"])
        print("Validated attention candidates:")
        for candidate in gpu["validated_attention_candidates"]:
            print(f"  - {candidate}")
    print("\nNOTES")
    for note in report["recommendations"]:
        print(f"- {note}")
    print("\nhttps://solricks.com")

def main():
    p = argparse.ArgumentParser(description="SOLRICKS AI Runtime — GPU diagnostics")
    selection = p.add_mutually_exclusive_group()
    selection.add_argument("--device", type=int, default=0, help="CUDA logical device index")
    selection.add_argument("--all-devices", action="store_true", help="Test each visible CUDA GPU")
    p.add_argument("--timeout", type=int, default=180, help="Seconds per isolated probe")
    p.add_argument("--probe-unlisted", action="store_true",
                   help="Experimental: attempt SageAttention APIs outside default SM routes")
    p.add_argument("--json", action="store_true", help="Print JSON only")
    p.add_argument("--output", type=Path, help="Also save report as JSON")
    p.add_argument("--version", action="version", version=VERSION)
    p.add_argument("--_worker", choices=["inventory", *TESTS], help=argparse.SUPPRESS)
    p.add_argument("--_result", type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.timeout < 1 or args.device < 0:
        p.error("timeout must be positive and device must be non-negative")
    if args._worker:
        if args._result is None:
            p.error("Missing worker result path")
        try:
            data = worker(args._worker, args.device, args.probe_unlisted)
        except Exception as exc:
            data = result("FAIL", f"{type(exc).__name__}: {exc}")
        args._result.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
        return 0
    report = diagnostics(args.device, args.all_devices, args.timeout, args.probe_unlisted)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        print_report(report)
    return report["exit_code"]

if __name__ == "__main__":
    sys.exit(main())
