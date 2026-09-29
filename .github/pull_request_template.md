# Pull Request Checklist

## Architecture health
- [ ] I read the canonical rules in `docs/dev/architecture_health.md` relevant to this change.
- [ ] I ran the local health command: `python tools/health.py`.
- [ ] I updated `docs/dev/architecture_health.md` if I changed guardrails.

## Public API & dependencies
- [ ] Public API changes are intentional; snapshot updated via `python tools/update_public_api_snapshot.py` if needed.
- [ ] No optional/heavy dependencies were added to core `[project].dependencies`.

## Behavior & tests
- [ ] Runtime behavior is unchanged (or explicitly documented as a bugfix).
- [ ] Tests pass locally (`pytest -q`).

## Reports/retention
- [ ] Raw audit/Goal evidence is external or a CI artifact; only durable conclusions are added to maintained docs.
- [ ] Generated output follows `docs/dev/repository_hygiene.md` and `python tools/check_repository_hygiene.py` passes.
