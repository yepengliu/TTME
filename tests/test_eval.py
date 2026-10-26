import errno
import json
import os
import time

import numpy as np
import pytest

from ttt import tasks
from ttt.eval import sandbox
from ttt.eval import score as score_mod
from ttt.eval.local import LocalEvaluator, evaluate_program
from ttt.eval.sandbox import run_program
from ttt.types import State


def parent(construction=None):
    return State(id="p", timestep=0, value=None, code="", construction=construction)


def test_success_converts_result_to_json():
    program = ("import numpy as np\n"
               "def main():\n"
               "    print('working')\n"
               "    return (np.arange(3, dtype=np.float32) / 2, np.float64(0.25), np.int64(7), [1, 2])\n")
    result, stdout, error = run_program(program, "main", 30, 1)
    assert error is None and stdout == "working\n"
    envelope = json.loads(result)
    assert envelope["value"] == [[0.0, 0.5, 1.0], 0.25, 7, [1, 2]]
    assert envelope["signature"] == {"type_name": "builtins.tuple", "is_list": False,
                                     "plain_numbers": False}


def test_raising_program_reports_error_and_traceback():
    program = "def main():\n    raise ValueError('boom')\n"
    result, stdout, error = run_program(program, "main", 30, 1)
    assert result is None and error == "Program execution failed: boom"
    assert "Traceback" in stdout and "ValueError: boom" in stdout


def test_missing_entrypoint_and_exit_without_result():
    _, _, error = run_program("x = 1\n", "main", 30, 1)
    assert error.startswith("Program execution failed:") and "main" in error
    _, _, error = run_program("import os\ndef main():\n    os._exit(0)\n", "main", 30, 1)
    assert error == "Program produced no result file; exit code 0."


def test_exit_status_explains_a_missing_result():
    _, _, error = run_program("import os\ndef main():\n    os._exit(3)\n", "main", 30, 1)
    assert error == "Program produced no result file; exit code 3."
    program = "import os, signal\ndef main():\n    os.kill(os.getpid(), signal.SIGKILL)\n"
    _, _, error = run_program(program, "main", 30, 1)
    assert error == "Program produced no result file; killed by signal 9 (SIGKILL, out of memory?)."
    assert sandbox.describe_exit(-15) == "killed by signal 15 (SIGTERM)"


def test_non_json_result_is_an_error():
    _, _, error = run_program("def main():\n    return {1, 2}\n", "main", 30, 1)
    assert error.startswith("Result is not JSON-serialisable:")


def test_timeout_keeps_partial_stdout_and_kills_the_group():
    program = ("import subprocess, time\n"
               "def main():\n"
               "    child = subprocess.Popen(['sleep', '60'])\n"
               "    print('child', child.pid)\n"
               "    print('tick')\n"
               "    time.sleep(60)\n")
    t0 = time.monotonic()
    result, stdout, error = run_program(program, "main", 1.5, 1)
    assert time.monotonic() - t0 < 10
    assert result is None and error == "Evaluation timed out after 1.5s."
    assert "tick" in stdout
    pid = int(stdout.split("child ")[1].split()[0])
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.parametrize("hang", [False, True], ids=["returns", "times-out"])
def test_descendants_in_new_sessions_are_killed(hang):
    program = ("import subprocess, time\n"
               "def main():\n"
               "    child = subprocess.Popen(['sleep', '60'], start_new_session=True)\n"
               "    print('child', child.pid)\n"
               f"    time.sleep({60 if hang else 0})\n"
               "    return [1]\n")
    result, stdout, error = run_program(program, "main", 1.5, 1)
    assert (result is None) is hang and (error is not None) is hang
    pid = int(stdout.split("child ")[1].split()[0])
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


ENVELOPE_JSON = ('{"value": [1], "signature": {"type_name": "builtins.list", "is_list": true, '
                 '"plain_numbers": true}}')


def test_oversized_linked_or_special_results_are_refused(monkeypatch):
    monkeypatch.setattr(sandbox, "RESULT_LIMIT_BYTES", 1000)
    result, _, error = run_program("def main():\n    return [1.0] * 1000\n", "main", 30, 1)
    assert result is None and error == "result.json exceeds the 1000-byte limit; refused."
    plant = ("import os\n"
             "def main():\n"
             f"    open('real.json', 'w').write({ENVELOPE_JSON!r})\n"
             "    PLANT\n"
             "    os._exit(0)\n")
    linked = plant.replace("PLANT", "os.symlink('real.json', 'result.json')")
    result, _, error = run_program(linked, "main", 30, 1)
    assert result is None and error == "result.json is a symlink; refused."
    result, _, error = run_program(plant.replace("PLANT", "os.mkfifo('result.json')"), "main", 30, 1)
    assert result is None and error == "result.json is not a regular file; refused."


def test_child_environment_holds_only_the_sandbox_variables(monkeypatch):
    # Guarantees only the child's own environment; the parent's stays readable by the same user (/proc).
    monkeypatch.setenv("TTT_PARENT_ONLY", "x")
    program = "import os\ndef main():\n    return dict(os.environ)\n"
    result, _, error = run_program(program, "main", 30, 3)
    env = json.loads(result)["value"]
    # The interpreter itself may add LC_CTYPE (PEP 538 locale coercion).
    assert error is None and set(env) - {"LC_CTYPE"} == {"PATH", "HOME", "TMPDIR", "PYTHONUNBUFFERED",
                                                          *sandbox.THREAD_VARS}
    assert env["OMP_NUM_THREADS"] == env["OPENBLAS_NUM_THREADS"] == env["MKL_NUM_THREADS"] == "3"
    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["HOME"].endswith("/home") and env["TMPDIR"].endswith("/tmp")
    assert os.path.dirname(env["HOME"]) == os.path.dirname(env["TMPDIR"])
    assert not os.path.exists(env["HOME"])


def test_process_pool_is_capped_and_program_runs_once():
    program = ("from concurrent.futures import ProcessPoolExecutor\n"
               "def init(tag):\n"
               "    print('init', tag)\n"
               "def square(x):\n"
               "    return x * x\n"
               "def main():\n"
               "    print('entry')\n"
               "    with ProcessPoolExecutor(16, initializer=init, initargs=('t',)) as ex:\n"
               "        values = list(ex.map(square, range(6)))\n"
               "        workers = ex._max_workers\n"
               "    return [values, workers]\n")
    result, stdout, error = run_program(program, "main", 60, 2)
    assert error is None and json.loads(result)["value"] == [[0, 1, 4, 9, 16, 25], 2]
    assert stdout.count("entry") == 1 and "bootstrapping" not in stdout


def test_local_evaluator_scores_ac1():
    task = tasks.get("ac1")
    code = "def propose_candidate():\n    print('hi')\n    return [1.0] * 1200\n"
    result = LocalEvaluator(1).evaluate("ac1", task.assemble(code, parent([0.5] * 1000)), 30, 1)
    assert result.ok and result.raw_score == 2.0 and result.stdout == "hi\n"
    assert result.construction == [1.0] * 1200 and result.error is None


def test_scoring_runs_outside_the_candidate_process():
    task = tasks.get("ac1")
    code = ("import numpy\n"
            "numpy.convolve = lambda a, b: numpy.zeros(len(a) + len(b) - 1)\n"
            "def evaluate_sequence(sequence):\n"
            "    return 0.0\n"
            "def propose_candidate():\n"
            "    seq = [1.0, 0.5, 2.0, 0.0] * 300\n"
            "    print('in-process score', evaluate_sequence(seq))\n"
            "    return seq\n")
    result = evaluate_program("ac1", task.assemble(code, parent()), 30, 1)
    assert "in-process score 0.0" in result.stdout
    assert result.ok and result.raw_score == tasks.score_sequence("ac1", [1.0, 0.5, 2.0, 0.0] * 300)


def test_erdos_end_to_end_ignores_reported_c5():
    task = tasks.get("erdos")
    code = ("def run(seed=42, budget_s=1000, **kwargs):\n"
            "    h = initial_h_values.copy()\n"
            "    corr = np.correlate(h, 1.0 - h, mode='full') * (2.0 / len(h))\n"
            "    return h, float(np.max(corr)) - 5e-5, len(h)\n")
    h = list(np.full(50, 0.5))
    result = evaluate_program("erdos", task.assemble(code, parent(h)), 30, 1)
    assert result.ok and result.raw_score == tasks.score_sequence("erdos", h)
    assert result.construction == h


def test_failed_run_is_not_scored():
    result = evaluate_program("ac2", "def construct_function():\n    raise KeyError('k')\n", 30, 1)
    assert not result.ok and result.raw_score is None and "KeyError" in result.stdout
    assert not result.infra_error


def test_candidate_timeouts_and_crashes_are_not_infrastructure_errors():
    hang = evaluate_program("ac2", "import time\ndef construct_function():\n    time.sleep(60)\n", 1, 1)
    assert hang.error == "Evaluation timed out after 1s." and not hang.infra_error
    crash = evaluate_program("ac2", "import os\ndef construct_function():\n    os._exit(7)\n", 30, 1)
    assert crash.error.endswith("exit code 7.") and not crash.infra_error


def test_sandbox_start_failures_are_infrastructure_errors(monkeypatch):
    def no_space(*args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(sandbox.tempfile, "mkdtemp", no_space)
    result = evaluate_program("ac1", "def propose_candidate():\n    return [1.0]\n", 30, 1)
    assert not result.ok and result.infra_error and "No space left" in result.error
    monkeypatch.undo()

    def no_fork(*args, **kwargs):
        raise OSError(errno.EAGAIN, "Resource temporarily unavailable")

    monkeypatch.setattr(sandbox.subprocess, "Popen", no_fork)
    result = evaluate_program("ac1", "def propose_candidate():\n    return [1.0]\n", 30, 1)
    assert not result.ok and result.infra_error and result.error.startswith("eval_sandbox_error:")
    with pytest.raises(sandbox.SandboxError):
        run_program("def main():\n    return 1\n", "main", 30, 1)


AC_RETURNS = [
    ("[1.0, 0.5, 2.0] * 400", True),
    ("list(np.array([1.0, 0.5, 2.0] * 400))", True),
    ("np.array([1.0, 0.5, 2.0] * 400)", False),
    ("tuple([1.0, 0.5, 2.0] * 400)", False),
    ("list(np.array([1, 2, 3] * 400, dtype=np.int64))", False),
    ("list(np.array([1.0, 0.5, 2.0] * 400, dtype=np.float32))", False),
    ("[1, 2, 3] * 400", True),
]


@pytest.mark.parametrize("name", ["ac1", "ac2"])
@pytest.mark.parametrize("expr,valid", AC_RETURNS, ids=[e for e, _ in AC_RETURNS])
def test_ac_validity_follows_the_returned_python_types(name, expr, valid):
    task = tasks.get(name)
    code = f"def {task.entrypoint}():\n    return {expr}\n"
    result = evaluate_program(name, task.assemble(code, parent()), 30, 1)
    assert result.ok is valid
    if valid:
        plain = [float(x) for x in eval(expr, {"np": np})]
        assert result.raw_score == tasks.score_sequence(name, plain)
    else:
        assert result.error == "Invalid solution." and result.raw_score is None


def envelope(value, is_list=True, plain=True):
    return json.dumps({"value": value, "signature": {"type_name": "builtins.list",
                                                     "is_list": is_list, "plain_numbers": plain}})


def test_score_result_failure_paths():
    assert not score_mod.score_result("ac1", "not json").ok
    assert score_mod.score_result("ac1", "not json").error.startswith("Scoring failed:")
    assert score_mod.score_result("ac1", json.dumps([1.0] * 10)).error.startswith("Scoring failed:")
    slow = score_mod.score_result("ac1", envelope([1.0] * 10), timeout_s=0.001)
    assert slow.error == "Scoring timed out after 0.001s."
    good = score_mod.score_result("ac2", envelope([1.0, 1.0]))
    assert good.ok and good.raw_score == tasks.score_sequence("ac2", [1.0, 1.0])
    assert not score_mod.score_result("ac2", envelope([1.0, 1.0], plain=False)).ok
    assert not slow.infra_error and not score_mod.score_result("ac1", "not json").infra_error


def test_scorer_start_failure_is_an_infrastructure_error(monkeypatch):
    def no_fork(*args, **kwargs):
        raise OSError(errno.EAGAIN, "Resource temporarily unavailable")

    monkeypatch.setattr(score_mod.subprocess, "run", no_fork)
    result = score_mod.score_result("ac2", envelope([1.0, 1.0]))
    assert not result.ok and result.infra_error and "cannot start the scorer" in result.error


def test_decode_result_validates_the_envelope():
    returned = score_mod.decode_result(envelope([1.0], is_list=False))
    assert returned.value == [1.0] and returned.is_list is False and returned.plain_numbers is True
    for bad in ("[1.0]", json.dumps({"value": [1.0]}),
                json.dumps({"value": [1.0], "signature": {"is_list": True}})):
        with pytest.raises(ValueError):
            score_mod.decode_result(bad)


def test_result_dict_round_trip_and_validation():
    result = tasks.get("ac1").score([1.0] * 3)
    assert score_mod.result_from_dict(score_mod.result_to_dict(result)) == result
    infra = score_mod.failed("eval_sandbox_error: x", infra_error=True)
    assert score_mod.result_from_dict(score_mod.result_to_dict(infra)) == infra
    for bad in (None, {"ok": 1, "stdout": ""}, {"ok": True, "stdout": "", "raw_score": "1"},
                {"ok": True, "stdout": "", "construction": 3}, {"ok": False, "stdout": 1},
                {"ok": False, "stdout": "", "infra_error": "yes"}):
        with pytest.raises(ValueError):
            score_mod.result_from_dict(bad)


def test_local_evaluator_validates_workers():
    assert LocalEvaluator(3).capacity == 3
    with pytest.raises(ValueError):
        LocalEvaluator(0)
