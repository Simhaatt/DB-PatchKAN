from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "02_colab_a100_stage1b_stage2_selection.ipynb"


def test_stage1b2_notebook_is_parseable_conditional_and_test_locked() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [cell for cell in payload["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) == 8
    sources = ["".join(cell["source"]) for cell in code_cells]
    for source in sources:
        ast.parse(source)
        assert "print(" in source or "update(" in source

    combined = "\n".join(sources)
    for required in (
        "DBPatchKAN_Stage1B2_Drive_Input.zip",
        "DBPatchKAN_stage1_reduced_seed42_ALL_MODELS_RESULTS.zip",
        "segments_train.npz",
        "segments_validation.npz",
        "gpu_tensor_split_manifest.json",
        "stage1b_formulation_selection.json",
        "stage2_seed_results.csv",
        "STAGE1B2_FINAL_SELECTION.json",
    ):
        assert required in combined

    assert "segments_test_UNTOUCHED.npz" not in combined
    assert "dbpatchkan.training.evaluate" not in combined
    assert "STAGE2_SEEDS = [42, 52, 62, 72, 82]" in combined
    assert "SELECTED_FORMULATION" in combined
    assert "SELECTED_HEAD" in combined
    assert "files.download(str(local_zip))" in sources[-1]
