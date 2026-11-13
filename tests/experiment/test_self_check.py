from vamos.experiment.diagnostics import self_check


def test_self_check_runs_numpy_only():
    results = self_check.run_self_check(verbose=False)
    assert any(r.name == "nsgaii-numpy" and r.status == "ok" for r in results)


def test_self_check_main_respects_quiet_argv(monkeypatch):
    calls: list[bool] = []

    def fake_run_self_check(*, verbose: bool):
        calls.append(verbose)
        return [self_check.CheckResult(name="nsgaii-numpy", status="ok")]

    monkeypatch.setattr(self_check, "run_self_check", fake_run_self_check)

    assert self_check.main(["--quiet"]) == 0
    assert calls == [False]


def test_self_check_reports_one_line_per_check(monkeypatch, caplog):
    import logging

    with caplog.at_level(logging.INFO):
        assert self_check.main([]) == 0
    lines = [record.getMessage() for record in caplog.records]
    names = [r.name for r in self_check.run_self_check(verbose=False)]
    for name in names:
        assert sum(line.startswith(f"[self-check] {name}:") for line in lines) == 1, (name, lines)
    # The runs' own summaries (problem, algorithm, backend...) are not printed.
    assert not any(line.startswith("Algorithm:") for line in lines)


def test_self_check_reports_a_numpy_failure_without_a_traceback(monkeypatch, capsys):
    def failing_check(engine, *, output_root):
        return self_check.CheckResult(name=f"nsgaii-{engine}", status="failed", detail="synthetic failure")

    monkeypatch.setattr(self_check, "_run_backend_check", failing_check)
    assert self_check.main([]) == 1
