"""Teacher quantization is named from the loaded layers, not from a label."""

from __future__ import annotations

import pytest
from data.build.teacher_quant_probe import classify_quantization


@pytest.mark.parametrize(
    ("classes", "method", "expected"),
    [
        ({"Linear4bit": 352, "Linear": 145}, "bitsandbytes", "bnb-nf4"),
        ({"Mxfp4GptOssExperts": 24, "Linear": 97}, "mxfp4", "mxfp4"),
        ({"Linear": 200}, None, "bf16/unquantized"),
        ({"Linear": 200}, "mxfp4", "dequantized from mxfp4"),
        ({}, None, "unverified"),
    ],
)
def test_classify_quantization(classes, method, expected) -> None:
    assert classify_quantization(classes, method) == expected


def test_an_mxfp4_teacher_without_kernels_is_refused(monkeypatch) -> None:
    import importlib.util
    from types import SimpleNamespace

    from data.build.teach_run import require_quantization_support

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a: None if name == "kernels" else real(name, *a)
    )
    with pytest.raises(RuntimeError, match="uv sync --extra teachers"):
        require_quantization_support(SimpleNamespace(quantization_config={"quant_method": "mxfp4"}))
    require_quantization_support(
        SimpleNamespace(quantization_config={"quant_method": "bitsandbytes"})
    )
    require_quantization_support(SimpleNamespace(quantization_config=None))
