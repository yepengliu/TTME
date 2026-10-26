import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ttt import tasks
from ttt.prompts import load
from ttt.tasks import Returned, ac, erdos
from ttt.tasks.common import hoist_future
from ttt.types import State

ROOT = Path(__file__).resolve().parents[1]
RESULTS = sorted((ROOT / "results").glob("*/sequence.json"))

TEMPLATE_SHA256 = {
    "ac1/initial_program.txt": "e941d029608c6ea86938297a40745bedc497e5b5fce0ca256b0e310704018c14",
    "ac1/literature.txt": "d4dc04ac543ab200effdaefab6e5c87ced5914b4c40b0a91f504104a9941d1f6",
    "ac1/question.txt": "4b9ed94d380cb0630637ca93be075c30a8bd7d91f86b99e1593347c97553ee5e",
    "ac1/shown_verifier.txt": "5737be1aa6417150827e58fdd374d240c22492af10ed532f9ad4ba82097082a4",
    "ac1/verifier.txt": "c221a8956e694ae52b978c0b9b36c8215a995e6846e12b5d51872fc5dee5b8fd",
    "ac2/initial_program.txt": "c346c1490264d0ae9c155c7a9fc9c630a397859d886afa5004ea54d049c23f71",
    "ac2/literature.txt": "bba34e9da4c5bb1649604881cca74756827860d54bb0b2db6b4a92d6b41732ab",
    "ac2/question.txt": "46c0f33b2d677906822947442d087390ad089b7366b2c52e21d835a2ae67ff8c",
    "ac2/shown_verifier.txt": "bbdfc60d65cab88a93e80ed1136c70dd549a02b130473710d4ef687603744884",
    "ac2/verifier.txt": "df0d837103d169464de8287dcf14ebdc8133639649bddedf1874dd6dcf2f633b",
    "erdos/code_section.txt": "6467c5445b76692768154eb443ff7fe6d5d1e520fc4fd4007df941486a07b3e4",
    "erdos/construction_section.txt": "60d081d76130aaa8f962f687837ca864981fd5d3a1e9ce7bfcd9767ca17a1c6f",
    "erdos/question.txt": "450e2c7f29914e194d9dadccbeda9a55e6b0c42f6fb11099e0fe1a0e8757d550",
    "erdos/verifier.txt": "0e8f31cfa416ede25a82dd35a6edcc7d1468d53045aa7e98371948102222194f",
}


def state(**kw):
    base = dict(id="s", timestep=3, value=None, code="", construction=None)
    base.update(kw)
    return State(**base)


def erdos_c5(h):
    h = np.asarray(h, dtype=np.float64)
    return float(np.max(np.correlate(h, 1.0 - h, mode="full") * (2.0 / len(h))))


def test_registry():
    assert tasks.TASK_NAMES == ("ac1", "ac2", "erdos")
    assert [tasks.get(n).maximize for n in tasks.TASK_NAMES] == [False, True, False]
    assert [tasks.get(n).entrypoint for n in tasks.TASK_NAMES] == [
        "propose_candidate", "construct_function", "run"]
    with pytest.raises(ValueError):
        tasks.get("ac3")


@pytest.mark.parametrize("name", sorted(TEMPLATE_SHA256))
def test_task_text_is_unchanged(name):
    data = load(f"tasks/{name}").encode("utf-8")
    assert hashlib.sha256(data).hexdigest() == TEMPLATE_SHA256[name]


@pytest.mark.parametrize("path", RESULTS, ids=[p.parent.name for p in RESULTS])
def test_stored_results_rescore_within_numeric_tolerance(path):
    record = json.loads(path.read_text())
    assert record["task"] == path.parent.name
    assert len(record["sequence"]) == record["n"]
    # Different BLAS builds can change the final bits of long convolutions.
    assert tasks.score_sequence(record["task"], record["sequence"]) == pytest.approx(
        record["score"], rel=1e-12, abs=1e-12)


def test_ac1_scores():
    task = tasks.get("ac1")
    assert task.score([1.0] * 1200).raw_score == 2.0
    clamped = task.score([-3.0, 1.0, 1.0, 5000.0])
    assert clamped.ok and clamped.raw_score == ac.verifier("ac1")([0.0, 1.0, 1.0, 1000.0])
    assert clamped.construction == [-3.0, 1.0, 1.0, 5000.0]
    for bad in ([], [float("nan")], [True, 1.0], ["1"], {"a": 1}, [0.001] * 5, None):
        res = task.score(bad)
        assert not res.ok and res.raw_score is None and res.error == "Invalid solution."


def test_ac2_scores():
    task = tasks.get("ac2")
    seq = [1.0, 0.5, 2.0, 0.0, 1.5]
    conv = np.convolve(seq, seq)
    y = np.concatenate(([0.0], conv, [0.0]))
    width = 1.0 / (len(conv) + 1)
    l2 = np.sum(width / 3 * (y[:-1] ** 2 + y[:-1] * y[1:] + y[1:] ** 2))
    expected = l2 / ((np.sum(conv) / (len(conv) + 1)) * np.max(conv))
    res = task.score(seq)
    assert res.ok and res.raw_score == pytest.approx(expected, rel=1e-12)
    assert res.construction == seq
    for bad in ([], [-1.0, 0.0], [float("inf")], [1, "2"], (1.0, 2.0)):
        assert not task.score(bad).ok


def test_ac_returned_record_types_decide_validity():
    for name in ("ac1", "ac2"):
        task = tasks.get(name)
        seq = [1.0, 0.5, 2.0] * 400
        good = task.score(Returned(seq, "builtins.list", is_list=True, plain_numbers=True))
        assert good.ok and good.raw_score == task.score(seq).raw_score and good.construction == seq
        for is_list, plain in ((False, True), (True, False), (False, False)):
            res = task.score(Returned(seq, "numpy.ndarray", is_list=is_list, plain_numbers=plain))
            assert not res.ok and res.error == "Invalid solution."
        assert not task.score(np.array(seq)).ok and not task.score(tuple(seq)).ok
        assert task.score(list(np.array(seq))).ok
        assert not task.score(list(np.array(seq, dtype=np.float32))).ok


def test_erdos_scores_returned_record_by_value():
    h = [1.0] * 20 + [0.0] * 20
    returned = Returned([h, 0.0, 40], "builtins.tuple", is_list=False, plain_numbers=False)
    assert tasks.get("erdos").score(returned) == tasks.get("erdos").score([h, 0.0, 40])


def test_erdos_scores_recomputed_c5():
    task = tasks.get("erdos")
    h = [1.0] * 20 + [0.0] * 20
    true = erdos_c5(h)
    res = task.score([h, true, 40])
    assert res.ok and res.raw_score == true and res.construction == h


def test_erdos_underreported_c5_does_not_improve_score():
    task = tasks.get("erdos")
    h = list(np.random.default_rng(0).uniform(0.2, 0.8, 60))
    h_norm = erdos.normalised_h(h, 60)
    true = erdos_c5(h_norm)
    res = task.score([h, true - 5e-5, 60])
    assert res.ok and res.raw_score == true
    assert task.score([h, 123.0, 60]).raw_score == true


def test_erdos_normalisation_and_invalid_cases():
    task = tasks.get("erdos")
    res = task.score([[0.25] * 10, 0.0, 10])
    assert res.ok and res.construction == [0.5] * 10
    assert res.raw_score == erdos_c5([0.5] * 10)
    bad = [
        [[0.5] * 10, 0.1, 11],
        [[[0.5] * 2] * 5, 0.1, 5],
        [[0.5] * 9 + [float("nan")], 0.1, 10],
        [[0.5] * 9 + [1.5], 0.1, 10],
        [[1.0] * 4 + [0.0] * 6, 0.1, 10],
        [[0.0] * 10, 0.1, 10],
        [[], 0.1, 0],
        [[0.5] * 10, 10],
        "abc",
        None,
    ]
    for result in bad:
        out = task.score(result)
        assert not out.ok and out.error == "Invalid solution." and out.construction is None


def test_erdos_stored_construction_rescores_to_its_score():
    task = tasks.get("erdos")
    for seed in range(20):
        h = list(np.random.default_rng(seed).uniform(0.05, 0.6, 200))
        res = task.score([h, 0.0, 200])
        assert res.ok
        assert tasks.score_sequence("erdos", res.construction) == res.raw_score


def test_score_sequence_rejects_invalid():
    with pytest.raises(ValueError):
        tasks.score_sequence("ac1", [])
    with pytest.raises(ValueError):
        tasks.score_sequence("ac2", [0.0, 0.0])
    with pytest.raises(ValueError):
        tasks.score_sequence("erdos", [2.0, 0.0])
    with pytest.raises(ValueError):
        tasks.score_sequence("ac3", [1.0])


def test_admissible():
    for name in ("ac1", "ac2"):
        task = tasks.get(name)
        assert [task.admissible([0.0] * n) for n in (999, 1000, 100000, 100001)] == [
            False, True, True, False]
    task = tasks.get("erdos")
    assert task.admissible([0.5] * 1000) and not task.admissible([0.5] * 1001)


def test_initial_states_follow_rng():
    for name in tasks.TASK_NAMES:
        task = tasks.get(name)
        a = task.initial_state(np.random.default_rng(5))
        b = task.initial_state(np.random.default_rng(5))
        assert (a.id, a.construction, a.value, a.code) == (b.id, b.construction, b.value, b.code)
        assert a.timestep == -1 and a.parent_values == [] and a.observation == ""
    ac1 = tasks.get("ac1").initial_state(np.random.default_rng(1))
    assert 1000 <= len(ac1.construction) < 8000 and len(set(ac1.construction)) == 1
    assert ac1.value == -ac.sequence_value("ac1", ac1.construction)
    assert ac1.code.startswith("```python\n") and ac1.code.endswith("\n```")
    ac2 = tasks.get("ac2").initial_state(np.random.default_rng(1))
    assert ac2.value == ac.sequence_value("ac2", ac2.construction)
    er = tasks.get("erdos").initial_state(np.random.default_rng(1))
    assert 40 <= len(er.construction) < 100 and er.code == ""
    assert er.value == pytest.approx(-erdos_c5(er.construction), rel=1e-12)


def test_refresh_initial():
    task = tasks.get("ac2")
    s = task.initial_state(np.random.default_rng(1))
    s.construction_hash = "old"
    task.refresh_initial(s, np.random.default_rng(99))
    assert s.value == ac.sequence_value("ac2", s.construction) and s.construction_hash is None
    er = tasks.get("erdos")
    s = er.initial_state(np.random.default_rng(1))
    before = list(s.construction)
    er.refresh_initial(s, np.random.default_rng(99))
    assert s.construction == before


def test_question_memory_placement():
    parent = state(code="```python\nx = 1\n```", value=-1.6, parent_values=[-1.7],
                   construction=[0.5] * 1500, observation="best 1.6")
    q = tasks.get("ac1").question(parent, "MEMORY BLOCK")
    assert "1.700000 -> 1.600000" in q and "Current gap: 0.097000" in q
    assert "--- End Output ---\n\nMEMORY BLOCK\nLength of the construction: 1500\n" in q
    assert tasks.get("ac1").question(parent, None) == tasks.get("ac1").question(parent, "")
    assert tasks.get("ac2").question(parent, None).startswith("'Act as an expert")
    eq = tasks.get("erdos").question(state(value=-0.39, construction=[0.5] * 50), "MEM")
    assert "(lower is better): 0.390000\n" in eq
    assert "rewarded.\n\nMEM\n\nYou may want to start your search" in eq
    assert eq.endswith("(n=50 samples).\nYou are encouraged to explore solutions that use other "
                       "starting points to prevent getting stuck in a local optimum.\n\n"
                       "Write code to optimize this construction.\n")


DIRECTION = {"ac1": ("upper bound", "lower is better", "higher is better"),
             "ac2": ("lower bound", "higher is better", "lower is better"),
             "erdos": ("C₅ bound", "lower is better", "higher is better")}
MID_RUN = {"ac1": (-1.6, -1.7, 1500), "ac2": (0.9, 0.89, 1500), "erdos": (-0.385, -0.39, 50)}


@pytest.mark.parametrize("name", tasks.TASK_NAMES)
def test_question_states_task_direction(name):
    task = tasks.get(name)
    metric, want, other = DIRECTION[name]
    value, parent_value, n = MID_RUN[name]
    mid = state(code="```python\nx = 1\n```", value=value, parent_values=[parent_value],
                construction=[0.5] * n, observation="done")
    for parent, line in (
            (task.initial_state(np.random.default_rng(3)), f"\nCurrent {metric} ({want}): "),
            (mid, f"\nHere is the {metric} before and after running the code above ({want}): ")):
        q = task.question(parent, None)
        assert line in q
        assert other not in q


def test_question_truncates_long_observation():
    parent = state(value=0.9, construction=[1.0] * 1000, observation="x" * 600 + "END")
    q = tasks.get("ac2").question(parent, None)
    assert "\n\n\t\t ...(TRUNCATED)...\n" + "x" * 497 + "END\n--- End Output ---" in q


def test_future_import_is_hoisted_and_offset_maps_lines():
    parent = state(construction=[0.5, 0.25])
    code = ('"""Doc."""\nfrom __future__ import annotations\nimport math\n\n'
            'def run(seed: int = 42):\n    return math.pi\n')
    hoisted, body = hoist_future(code)
    assert hoisted == "from __future__ import annotations\n"
    for name in tasks.TASK_NAMES:
        task = tasks.get(name)
        assembled = task.assemble(code, parent)
        assert assembled.startswith(hoisted + "import numpy as np\n\n")
        compile(assembled, "program.py", "exec")
        offset = task.line_offset(code, parent)
        assert assembled.split("\n")[offset:] == body.split("\n")


def test_misplaced_future_import_is_left_alone():
    code = "import math\nfrom __future__ import annotations\n"
    assert hoist_future(code) == ("", code)
    assembled = tasks.get("ac1").assemble(code, state(construction=None))
    assert assembled.endswith("\n\n" + code) and "height_sequence_1 =" not in assembled


def test_assemble_injects_parent_construction():
    parent = state(construction=[0.5, 1, np.float64(0.125)])
    code = "def run():\n    return 1\n"
    assembled = tasks.get("erdos").assemble(code, parent)
    assert "initial_h_values = np.array([0.5, 1, 0.125])\n\n" + code in assembled
    assert tasks.get("ac1").line_offset(code, parent) == tasks.get("ac1").assemble(
        code, parent).count("\n") - 2
    with pytest.raises(ValueError):
        tasks.get("ac1").assemble(code, None)
