import unittest
from unittest.mock import patch
import runtime

class RuntimeTests(unittest.TestCase):
    def test_ampere_fp8_not_default(self):
        self.assertNotIn((8, 6), runtime.SAGE_ROUTES["sage-fp8"])

    def test_ada_fp8(self):
        self.assertIn((8, 9), runtime.SAGE_ROUTES["sage-fp8"])
        self.assertEqual(runtime.sage_spec("sage-fp8", (8, 9))[1],
                         {"pv_accum_dtype": "fp32+fp16"})

    def test_blackwell_fp8(self):
        for sm in [(12, 0), (12, 1)]:
            self.assertIn(sm, runtime.SAGE_ROUTES["sage-fp8"])
            self.assertEqual(runtime.sage_spec("sage-fp8", sm)[1]["qk_quant_gran"], "per_warp")

    def test_hopper(self):
        self.assertIn((9, 0), runtime.SAGE_ROUTES["sage-sm90"])
        self.assertEqual(runtime.sage_spec("sage-sm90", (9, 0))[1]["pv_accum_dtype"], "fp32+fp32")

    def test_unknown_route(self):
        self.assertNotIn((10, 0), runtime.SAGE_ROUTES["sage-auto"])

    @patch("runtime.driver_info", return_value=[])
    @patch("runtime.run_worker", return_value={"status": "SKIP", "detail": "No torch"})
    def test_no_runtime(self, worker, driver):
        self.assertEqual(runtime.diagnostics(0)["exit_code"], 2)

    @patch("runtime.driver_info", return_value=[])
    @patch("runtime.run_worker")
    def test_all_devices_and_failure(self, worker, driver):
        def mock(task, *args, **kwargs):
            if task == "inventory":
                return {"status": "PASS", "detail": "OK", "data": {
                    "devices": [
                        {"index": 0, "sm": [8, 6], "cuda_free_gib": 20},
                        {"index": 1, "sm": [12, 0], "cuda_free_gib": 25},
                    ]}}
            return {"status": "FAIL" if task == "sage-fp8" else "PASS", "detail": "mock"}
        worker.side_effect = mock
        report = runtime.diagnostics(0, all_devices=True)
        self.assertEqual(len(report["gpus"]), 2)
        self.assertEqual(report["exit_code"], 1)
        self.assertEqual(next(t for t in report["gpus"][0]["tests"]
                              if t["id"] == "sage-fp8")["status"], "SKIP")
        self.assertEqual(next(t for t in report["gpus"][1]["tests"]
                              if t["id"] == "sage-fp8")["status"], "FAIL")

    @patch("runtime.subprocess.run", side_effect=runtime.subprocess.TimeoutExpired("worker", 1))
    def test_timeout(self, run):
        self.assertEqual(runtime.run_worker("cuda", timeout=1)["status"], "TIMEOUT")

    @patch("runtime.subprocess.run")
    def test_worker_crash(self, run):
        run.return_value = runtime.subprocess.CompletedProcess([], -11, "", "crash")
        self.assertEqual(runtime.run_worker("cuda")["status"], "FAIL")

if __name__ == "__main__":
    unittest.main()
