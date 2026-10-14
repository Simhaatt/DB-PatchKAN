from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "03_colab_a100_kan_explainability.ipynb"


def test_stage3_notebook_is_parseable_explainable_and_test_locked() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [cell for cell in payload["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) == 10
    sources = ["".join(cell["source"]) for cell in code_cells]
    for source in sources:
        ast.parse(source)
        assert "print(" in source or "update(" in source

    combined = "\n".join(sources)
    for required in (
        "DBPatchKAN_Stage3_Upload_Input.zip",
        "DBPatchKAN_stage1b2_full_selection_ALL_MODELS_RESULTS.zip",
        "concept_response_curves",
        "concept_integrated_gradients",
        "validation_occlusion_tables",
        "kan_pruning_sensitivity.csv",
        "kan_hidden=()",
        "DBPatchKAN_Stage3_Explainability.xlsx",
        "STAGE3_SUMMARY.json",
    ):
        assert required in combined

    assert "segments_test_UNTOUCHED.npz" not in combined
    assert "--test-data" not in combined
    assert "files.download(str(local_zip))" in sources[-1]
