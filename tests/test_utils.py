#!/usr/bin/env python
"""Tests for resolve_device CUDA diagnostic logging and hard-failure behavior."""

import logging
import sys
import unittest
from unittest.mock import MagicMock, patch


def _make_torch_mock(cuda_available: bool) -> MagicMock:
    torch_mock = MagicMock()
    torch_mock.cuda.is_available.return_value = cuda_available
    torch_mock.cuda.device_count.return_value = 1
    torch_mock.cuda.get_device_name.return_value = "Tesla T4"
    torch_mock.version.cuda = "12.1"
    torch_mock.backends.cuda.is_built.return_value = True
    # Make torch.device("cpu") return a simple string-like object
    torch_mock.device.side_effect = lambda d: d
    return torch_mock


class TestResolveDeviceDiagnostics(unittest.TestCase):
    """Verify that resolve_device emits the correct diagnostic logs."""

    def setUp(self):
        # Ensure utils is reloaded fresh with each test
        sys.modules.pop("filter_huggingface_vision.utils", None)
        sys.modules.pop("torch", None)

    def tearDown(self):
        sys.modules.pop("filter_huggingface_vision.utils", None)
        sys.modules.pop("torch", None)

    def test_cuda_available_logs_info(self):
        torch_mock = _make_torch_mock(cuda_available=True)
        with patch.dict("sys.modules", {"torch": torch_mock}):
            import filter_huggingface_vision.utils as utils_mod

            with self.assertLogs("filter_huggingface_vision.utils", level=logging.INFO) as cm:
                utils_mod.resolve_device("cuda")

        log_text = "\n".join(cm.output)
        self.assertIn("Requested device: cuda", log_text)
        self.assertIn("CUDA available: True", log_text)
        self.assertIn("Tesla T4", log_text)
        self.assertIn("12.1", log_text)

    def test_cuda_unavailable_raises_runtime_error(self):
        """Explicit cuda request with no CUDA must raise RuntimeError, not fall back silently."""
        torch_mock = _make_torch_mock(cuda_available=False)
        with patch.dict("sys.modules", {"torch": torch_mock}):
            import filter_huggingface_vision.utils as utils_mod

            with self.assertRaises(RuntimeError) as ctx:
                utils_mod.resolve_device("cuda")

        msg = str(ctx.exception)
        self.assertIn("FILTER_DEVICE='cuda'", msg)
        self.assertIn("CUDA is not available", msg)
        self.assertIn("12.1", msg)  # PyTorch CUDA version
        self.assertIn("nvidia-smi", msg)  # driver check suggestion

    def test_cpu_device_no_diagnostics(self):
        """Requesting CPU should not trigger any CUDA diagnostic logs."""
        torch_mock = _make_torch_mock(cuda_available=False)
        with patch.dict("sys.modules", {"torch": torch_mock}):
            import filter_huggingface_vision.utils as utils_mod

            with self.assertNoLogs("filter_huggingface_vision.utils", level=logging.INFO):
                utils_mod.resolve_device("cpu")

        # CUDA availability should never be checked for a CPU request
        torch_mock.cuda.is_available.assert_not_called()

    def test_cuda_integer_device_available(self):
        torch_mock = _make_torch_mock(cuda_available=True)
        with patch.dict("sys.modules", {"torch": torch_mock}):
            import filter_huggingface_vision.utils as utils_mod

            with self.assertLogs("filter_huggingface_vision.utils", level=logging.INFO) as cm:
                result = utils_mod.resolve_device(0)

        log_text = "\n".join(cm.output)
        self.assertIn("Requested device: cuda:0", log_text)
        self.assertIn("CUDA available: True", log_text)
        self.assertEqual(result, "cuda:0")

    def test_cuda_integer_device_unavailable_raises(self):
        """Explicit integer CUDA device with no CUDA must raise RuntimeError."""
        torch_mock = _make_torch_mock(cuda_available=False)
        with patch.dict("sys.modules", {"torch": torch_mock}):
            import filter_huggingface_vision.utils as utils_mod

            with self.assertRaises(RuntimeError) as ctx:
                utils_mod.resolve_device(0)

        msg = str(ctx.exception)
        self.assertIn("FILTER_DEVICE='cuda:0'", msg)
        self.assertIn("CUDA is not available", msg)
        self.assertIn("nvidia-smi", msg)

    def test_auto_device_cuda_available(self):
        """FILTER_DEVICE=auto with CUDA present should select cuda and log info."""
        torch_mock = _make_torch_mock(cuda_available=True)
        with patch.dict("sys.modules", {"torch": torch_mock}):
            import filter_huggingface_vision.utils as utils_mod

            with self.assertLogs("filter_huggingface_vision.utils", level=logging.INFO) as cm:
                result = utils_mod.resolve_device("auto")

        self.assertEqual(result, "cuda")
        log_text = "\n".join(cm.output)
        self.assertIn("auto", log_text)
        self.assertIn("cuda", log_text)

    def test_auto_device_cuda_unavailable_warns_and_falls_back(self):
        """FILTER_DEVICE=auto with no CUDA should warn and return cpu, not raise."""
        torch_mock = _make_torch_mock(cuda_available=False)
        with patch.dict("sys.modules", {"torch": torch_mock}):
            import filter_huggingface_vision.utils as utils_mod

            with self.assertLogs("filter_huggingface_vision.utils", level=logging.WARNING) as cm:
                result = utils_mod.resolve_device("auto")

        self.assertEqual(result, "cpu")
        log_text = "\n".join(cm.output)
        self.assertIn("falling back to CPU", log_text)
