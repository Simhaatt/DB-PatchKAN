from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "01_colab_a100_reduced_stage1.ipynb"


def test_stage1_notebook_is_parseable_progressive_and_test_locked() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [cell for cell in payload["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) == 9
    sources = ["".join(cell["source"]) for cell in code_cells]
    for source in sources:
        ast.parse(source)
        assert "print(" in source or "update(" in source

    combined = "\n".join(sources)
    for required in (
        "dbpatchkan_colab_code.zip",
        "segments_train.npz",
        "segments_validation.npz",
        "gpu_tensor_split_manifest.json",
    ):
        assert required in combined
    assert '"--search-profile", "reduced"' in combined
    assert "files.download(str(local_zip))" in sources[-1]
    assert "--test-data" not in combined
    assert "dbpatchkan.training.evaluate" not in combined
