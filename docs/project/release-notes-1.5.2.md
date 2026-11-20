# VAMOS 1.5.2 release notes

Released 2026-11-20.

VAMOS 1.5.2 is a patch release of [VAMOS 1.5.1](release-notes-1.5.1.md). It
adds reformulations of 23 RCM problems, whose variables decode into designs
that meet the original problems' equality and ordering constraints, and
evaluates RCM30–RCM35 faster, with the same values bit for bit. The
reformulations are an addition, which the
[semantic-versioning policy](stability-and-versioning.md#semantic-versioning-policy)
leaves to minor releases (patch releases fix defects); the maintainers chose
1.5.2 for them. No existing problem, algorithm, default, configuration or
result changes, and runs saved with VAMOS 1.5.1 verify and replay as before.
The stable 1.x API, the omitted-engine NumPy default, and the 1.0.0 run and
study schemas remain the compatibility baseline.

## Reformulated RCM problems

Standard evolutionary algorithms find no feasible design of many problems of
the CEC 2021 real-world constrained suite (Kumar et al., 2021): their
equalities (`|h| <= 1e-4`) leave thin feasible sets, and the ordering
constraints of RCM30–RCM35 leave constraint-dominance on a plateau. A
reformulation searches other variables, its genotype, and decodes each
genotype into a complete decision vector of the original problem that meets
those constraints by construction. 23 problems have one, registered as
`rcm22_reformulated`, `rcm24_reformulated`, `rcm28_reformulated` and
`rcm30_reformulated` to `rcm49_reformulated`; see
[Reformulated RCM problems](../reference/problems.md#reformulated-rcm-problems)
in the problems catalog, which now lists 303 registered problems.

- **Decoders.** RCM22, RCM24 and RCM28 decode in closed form. RCM30–RCM35
  decode into increasing angles, in three variants chosen with the problem
  option `variant`: `"cumulative"` by default, the cumulative-variable
  reformulation of Wang, Zhu, Nan and Ishibuchi (GECCO '25 Companion),
  `"sorted"` or `"level"`. RCM36–RCM49 solve their power flows by Newton's
  method from a flat start.
- **Values.** `F` and the inequality constraints are the original problem's at
  the decoded design: the original evaluates it, so they are equal bit for bit.
  A last constraint flags a failed decoding, so a row is feasible on the
  reformulation exactly when its decoded design is feasible on the original
  problem. Nothing in `F` or `G` is NaN.
- **Mapping back.** The instance holds the original problem as `original`.
  `decode(U)` returns the original's decision vectors, with a row of NaN where
  the decoding failed, and `evaluate_decoded(U, out)` evaluates as `evaluate`
  does and returns the designs of the same decoding.
- **Runs and fronts.** A run records the `variant` option and replays exactly.
  A reformulation resolves its original's reference front, since its
  objectives are the original's.

The problems are registered by name: `vamos.problems` exports no new class, so
the public API does not change.

## Faster RCM30–RCM35

The six SOPWM problems, RCM30–RCM35, evaluate about 30 times faster per call
with one row, and faster with many. The evaluation is vectorized over the
harmonics and the angles, with running sums that add the terms in jMetal's
order, so the values are the same bit for bit. Steady-state algorithms, such
as MOEA/D and SMS-EMOA, which evaluate one row per call, gain the most.

## Which results change

None. RCM30–RCM35 evaluate the same values bit for bit, only faster, so a run
of these problems gives the same result as with VAMOS 1.5.1. The new problems
add to what VAMOS resolves: every problem key and reference front that
resolved with VAMOS 1.5.1 resolves to the same problem or file.

## Dependencies

The core and optional dependencies do not change; the reformulations need only
NumPy.

## Known limitations

See [known limitations](known-limitations.md).
