# VAMOS-CONFIG design: parameter spaces as grammars

**Development after 1.2.1:** this is a design draft for review. Nothing in it is implemented,
and it does not describe the behaviour of any released version. The tuning spaces it discusses
are Experimental (see [Stability and versioning](../project/stability-and-versioning.md)).

VAMOS-Studio's design note
[Parameter spaces as grammars](https://github.com/vamos-optimization/VAMOS-Studio/blob/main/docs/design/parameter-spaces.md)
prototypes the model below on VAMOS 1.1.0's spaces. This draft takes its direction, applies it
to the spaces of `main`, and adds what VAMOS itself has to change.

## 1. Goal

A parameter space says how an algorithm can be configured. Today each built-in space is a Python
builder (`src/vamos/engine/tuning/racing/bridge_spaces_*.py`), and every consumer reads it in its
own way: the racing tuner and the model-based backends (`_model_backend_utils.py`,
`_irace_space.py`) translate its condition expressions, `config_from_assignment` turns an
assignment into a configuration, and VAMOS-Studio rebuilds a form from it.

VAMOS-CONFIG makes one model the source of truth for all of them:

- **a space is a strict grammar with defaults**, written in the vocabulary of the algorithm
  configurations;
- **a configuration is a derivation of that grammar**;
- **ConfigSpace, Optuna, irace, the derivation-tree encoding and the flat encoding are views**
  derived from the grammar, never edited by hand.

The goal is Evolver's automatic algorithm configuration in VAMOS: a configuration tuned with
any of the tools above, or published with Evolver, can be read, run, compared with the default
configuration and replayed.

## 2. Problems in the current spaces

Survey of `main` (`a6857d55`): 53 spaces, 98 distinct parameters with 1,038 occurrences, 314
conditional blocks of the form `parent == value`, and 67 free-form conditions.

| Id | Problem | Evidence |
| --- | --- | --- |
| P1 | **No defaults.** Parameters have ranges or choices but no default, so a tuner cannot start from the default configuration and Studio has to invent starting values. | Every space. |
| P2 | **Some defaults are not among the choices.** `initializer` and `repair` default to automatic choices that no space offers; NSGA-II's `scatter_base_size_factor` defaults to 0.25 in the bridge, which is not one of its six choices. | `apply_initializer`, `bridge_spaces_nsgaii.py`. Until `fix/smsemoa-tuning-selection`, no SMS-EMOA assignment reproduced `SMSEMOAConfig.default()`. |
| P3 | **Operators without a default for a tuned parameter.** `GaussianMutation` requires `sigma`, so `gaussian_sigma` has no value to start from; the bridge falls back to 1.0, outside the space's range [0.001, 0.5]. | `operators/impl/real/mutation.py`, `_bridge_assignment_shared.py`. |
| P4 | **Conditions that are not a tree.** `archive_prune_policy` depends on `archive_unbounded == False` inside `use_external_archive` (49 spaces); `mutation_eta` is shared by two mutations through an `or` expression (16); `crossover_prob` depends on `crossover != 'de'` (CCMO and CMOEA-MS). A parameter with two parents, or an inequality, has no place in a derivation tree. | `bridge_space_parts_discrete.py`, `bridge_spaces_ccmo.py`, `bridge_spaces_cmoeams.py`. |
| P5 | **More than one vocabulary.** `pm` and `polynomial` name the same mutation in different spaces; MOEA/D and SMS-EMOA tuned an absolute `mutation_prob` whose default (1/n) depends on the problem, while NSGA-II tunes `mutation_prob_factor`; the names differ from Evolver's (`neighbor_size` against `neighborhoodSize`, `undx` against `UNDC`). | `feat/evolver-space-ranges` moves MOEA/D and SMS-EMOA to the factor; `feat/evolver-parameter-names` adds Evolver's names. |

## 3. Proposed decisions

1. **Strict grammar.** A categorical parameter chooses between productions, one per value; a
   production holds the parameters that its value activates; numeric and boolean parameters are
   terminals. Every parameter has exactly one parent: a production, or a categorical node as one
   of its global sub-parameters. There are no condition expressions.
2. **Every parameter has a default, inside its domain**, and the defaults of a space form the
   default configuration of its algorithm and encoding: building them gives a configuration
   that runs exactly as `XConfig.default(n_var=..., n_obj=..., encoding=...)`. A test checks this
   for every shipped space by comparing seeded runs, not dictionaries, because equivalent
   configurations can be written differently (`("pm", {"prob": 1/n})` against
   `mutation_prob_factor: 1.0`).
3. **The vocabulary of the configurations.** Parameter and value names are those that the
   configuration builders accept, one name per operator, with Evolver's snake_case names as the
   canonical parameter names (see section 8). One function per algorithm turns a configuration of
   the grammar into its `XConfig`; `config_from_assignment` already plays that role.
4. **Problem-relative settings.** A space does not know the problem. Settings that depend on it
   are relative and resolved when the configuration is built: `mutation_prob_factor` gives
   `factor / n_var`; MOEA/D, NSGA-III and RVEA derive their population from `n_obj` and the weight
   vectors or reference directions, and their grammars say how a sampled `pop_size` is rounded.
5. **The grammar is data.** It serializes to YAML and JSON under a versioned schema,
   `vamos.parameter-space` 1.0.0. VAMOS ships its spaces as package resources (or builders that
   emit grammars) and users can load reduced or extended spaces, which the same validator checks.
6. **Views are derived, never the reverse.** ConfigSpace, Optuna, irace, the tree and flat
   encodings, the CLI's tuning output and Studio's forms come from the grammar without loss.

## 4. The grammar document

The document has a top level (`schema`, `schema_version`, `name`, `algorithm`, `encoding`,
`description`, and the ordered `parameters` that hang from the root) and parameter nodes:

- every node: `type` (`categorical`, `real`, `int`, `boolean`), `default`, and optionally
  `description`, `label` and `role` (VAMOS's tuning roles: `structural`, `operator`,
  `operator_rate`, `population`, `adaptive`);
- numeric nodes: `range` (`[low, high]`, both included) and optionally `log`;
- categorical nodes: `values` (a list of scalars, or a mapping from each value to its production,
  which may hold `conditional_parameters`) and optionally `global_sub_parameters`, active whatever
  the value.

A sketch of MOEA/D's real-coded variation in this form:

```yaml
variation:
  type: categorical
  default: differential_evolution
  values:
    differential_evolution:
      conditional_parameters:
        de_scheme: {type: categorical, default: rand_1_bin, values: [rand_1_bin, rand_1_exp, rand_2_bin, rand_2_exp]}
        de_crossover_rate: {type: real, range: [0.0, 1.0], default: 1.0}
        de_scale_factor: {type: real, range: [0.0, 1.0], default: 0.5}
    crossover_and_mutation:
      conditional_parameters:
        crossover:
          type: categorical
          default: sbx
          global_sub_parameters:
            crossover_probability: {type: real, range: [0.6, 1.0], default: 0.9}
          values:
            sbx:
              conditional_parameters:
                sbx_distribution_index: {type: real, range: [5.0, 400.0], default: 20.0}
            blx_alpha:
              conditional_parameters:
                blx_alpha_crossover_alpha: {type: real, range: [0.0, 1.0], default: 0.5}
```

The validator rejects a space that breaks any of these rules:

| Rule | Statement |
| --- | --- |
| G1 | Parameter names are unique in the space and match `^[a-z][a-z0-9_]*$`. |
| G2 | Each parameter has one parent; there are no expressions. A parameter that applies to some values of a node is duplicated under each of their productions, with its own name. |
| G3 | Every `default` belongs to its domain. |
| G4 | Ranges are ordered, `log` needs a positive lower end, and integer ranges have integer ends. |
| G5 | `values` is not empty and has no duplicates; a value with a production is a string. |
| G6 | `schema` and `schema_version` are known; `algorithm` and `encoding` exist in VAMOS. |
| G7 | The default configuration runs as `XConfig.default()` (decision 2). |

A space is identified in configurations and run manifests by `name`, `schema_version` and the
SHA-256 of its canonical JSON form (insertion order kept: it is the depth-first order of the
tree). This identity replaces the hand-bumped suffix of the persistent Optuna studies
(`__vamos_cli_final_population_hv_v3` on the tuning stack): a changed space changes its hash.

## 5. Configurations

A configuration is a derivation tree: every categorical node has a value, and under it hang its
global sub-parameters and the parameters of the chosen production. It is valid when it holds
exactly the active parameters, each inside its domain. Because the grammar is strict, the active
parameters listed by name in depth-first order determine the tree, so three serializations carry
the same information:

- **the canonical document**, `vamos.parameter-configuration` 1.0.0, with the space's identity
  and the ordered `values`;
- **the flat mapping** (the `values` alone), which Optuna, SMAC3 and irace exchange, and which
  `config_from_assignment` reads today;
- **the command-line string** of Evolver's `VAR_CONF.txt`, `--crossover sbx --sbx_distribution_index 20.0 ...`,
  and its Evolver form with camelCase names, `--crossover SBX --sbxDistributionIndex 20.0 ...`,
  through the name table of section 8.

Parsing any of them rebuilds the tree and validates it. A run tuned with VAMOS-CONFIG records the
configuration document next to the resolved specification, so a replay can say which space and
which point produced it.

## 6. Views derived from the grammar

| Consumer | View |
| --- | --- |
| VAMOS's evolutionary algorithms as meta-optimizers (main encoding) | The derivation tree. A random tree picks a production at every categorical node and samples every terminal; subtree crossover exchanges subtrees rooted at the same parameter; tree mutation changes one node and regrows a changed production. This needs a tree solution type and its operators, a new encoding next to `real`, `binary`, `integer`, `permutation` and `mixed`, which are part of the stable contract, so it needs its own design. |
| The same, secondary encoding | The flat encoding: one gene in [0, 1] per parameter, in depth-first order; decoding reads only the genes of active parameters, and a categorical gene of `k` values takes index `min(floor(u k), k - 1)`. Continuous meta-optimizers such as SMPSO use it. |
| SMAC3 and BOHB | A `ConfigurationSpace` whose hyperparameters carry `default_value`; a conditional parameter gets `EqualsCondition(child, parent, value)`, and a global sub-parameter of a non-root node gets `InCondition(child, node, all_values)`. |
| Optuna | Define-by-run: a depth-first walk calls `suggest_categorical`, `suggest_float` or `suggest_int` for every active node, and the default configuration is enqueued as the first trial. |
| irace | A parameters file with one line per parameter, a condition `\| parent == "value"` per conditional parameter and `\| node %in% c(...)` per global sub-parameter of a non-root node, and the default configuration in the initial configurations file. |
| VAMOS's racing tuner | The flat mapping, sampled by walking the tree, so that the active-parameter rule of today's `filter_active_config` becomes structural. |
| VAMOS-Studio | The tree itself: the form shows the active branch and starts from the defaults. |

Rewriting `_model_backend_utils.py` and `_irace_space.py` over the grammar removes their parsing
of condition expressions.

## 7. Migrating the current spaces

| Id | Change | Fixes | Scope on `main` |
| --- | --- | --- | --- |
| M1 | One categorical node `external_archive: none \| unbounded \| bounded`, with `archive_prune_policy` (and its parameters) under `bounded`, replacing `use_external_archive`, `archive_unbounded` and the expression on them. It is Evolver's `algorithmResult` and `archiveType`. | P4 | 49 spaces |
| M2 | `mutation_eta` duplicated as `polynomial_mutation_distribution_index` and `linked_polynomial_mutation_distribution_index`, as Evolver does. | P4 | 16 spaces |
| M3 | CCMO's and CMOEA-MS's `crossover_prob` moved under the productions of the GA crossovers (or a `variation` node as in M8). | P4 | 2 spaces |
| M4 | A default for every parameter: the configuration's default where it exists (`random` initializer and `clip` repair run as the automatic choices on real-coded problems, which a seeded comparison has to confirm per algorithm), `0.25` added to `scatter_base_size_factor`'s choices or the default moved to a choice, and a default added to every operator parameter that lacks one (`GaussianMutation.sigma` first). | P1, P2, P3 | about 40 parameters |
| M5 | Operator rates as global sub-parameters of their operator node (`crossover_probability` under `crossover`, `mutation_probability_factor` under `mutation`), so that a subtree crossover moves them with their operator. | Encoding quality | every space |
| M6 | One name per operator and parameter: `polynomial` everywhere, Evolver's snake_case names for parameters, and the current names accepted when reading old assignments. | P5 | every space |
| M7 | `mutation_probability_factor` instead of an absolute probability in every space, the discrete ones included (`bitflip`, the permutation and integer mutations). | P5 | the remaining binary, integer and permutation spaces |
| M8 | A `variation` node in MOEA/D's real-coded space, with `differential_evolution` and `crossover_and_mutation` productions, as Evolver's `MOEADDouble`. | DE schemes in section 9 | 2 spaces |

## 8. Names

`feat/evolver-parameter-names` adds a table between VAMOS's parameter names and Evolver's, in code
and in the tuning guide: VAMOS's current name, the canonical snake_case name (Evolver's name where
Evolver has the parameter, a descriptive one otherwise), and Evolver's camelCase name. With it,
the spaces use the canonical names, the current ones remain aliases, and configurations can be
exchanged with Evolver in both directions. The configuration classes keep their Stable field
names until VAMOS 2.0, so the table also maps canonical names to configuration fields.

The grammar adopts the canonical names as its vocabulary (decision 3). Two consequences:

- reading Evolver's own YAML spaces becomes a matter of the name table plus the value names
  (`SBX` against `sbx`, `UNDC` against `undc`, `tschebyscheff` against `tchebycheff`), so VAMOS and
  Evolver can be compared on the same space;
- VAMOS-Studio's renamed parameters (its M2 and M8) map onto canonical names, and the Studio can
  drop its local adapter.

## 9. What remains of the MOEA/D and SMS-EMOA comparison

`feat/evolver-space-ranges` covers the extensions that need no grammar (Evolver's neighbourhood
ranges, `pbi_theta` up to 200, `modified_tchebycheff`, linked polynomial mutation, the mutation
factor, `repair` and `initializer` in MOEA/D's space). These need the grammar, the engine, or both:

| Item | Proposal | Needs |
| --- | --- | --- |
| DE schemes | `de_scheme` with `rand_1_bin`, `rand_1_exp`, `rand_2_bin` and `rand_2_exp`, under M8's `differential_evolution` production; the same crossover set as NSGA-II's space under `crossover_and_mutation`, except PCX, UNDC and simplex, which need three parents while MOEA/D pairs them; SBX's index in [5, 400] as Evolver. | MOEA/D's `DifferentialCrossoverStep` implements rand/1/bin only; CMOEA-MS already has `rand_1_exp`. Rand/2 needs five parents from the neighbourhood. |
| Normalization in the aggregation | A global sub-parameter of `aggregation_function`, `normalize_objectives` (boolean) with `normalization_epsilon` in [1e-8, 25], as Evolver; candidates `augmented_tchebycheff` and `inverted_pbi` (`ipbi_theta` in [0.01, 200]). | Normalization by the ideal and nadir points in the aggregation functions and in the compiled neighbourhood kernel, which must keep its parity with NumPy. |
| Order of the subproblems | `subproblem_order: random_permutation \| cyclic`, Evolver's `subProblemIdGenerator`. | A cyclic order in MOEA/D's step; the steady-state kernels take the order as input. |
| Operators of SMS-EMOA and MOEA/D | The crossovers and mutations that VAMOS implements and NSGA-II's space already offers (whole arithmetic, Laplace, fuzzy recombination, BLX-αβ, Lévy flight, power law), and SBX's and polynomial mutation's indices up to 400, in both spaces; the Sobol and opposition-based initializers of NSGA-II's space. | Spaces only, once the names are settled (M6); the operators exist. |
| Binary MOEA/D and SMS-EMOA | The `bitflip` factor (M7); crossover probabilities as Evolver (MOEA/D [0.6, 0.9], SMS-EMOA [0, 1]); the same aggregation, archive and selection changes as the real-coded spaces. | Spaces only. |

## 10. Plan and compatibility

1. **Schema and validator** (`vamos.parameter-space` 1.0.0, rules G1–G7), with the configuration
   document and its three serializations. No behaviour changes.
2. **Shipped grammars**, generated from the current builders with M1–M7 applied, plus a
   drift-detection snapshot. The bridges read canonical names and keep the current ones as
   aliases, so stored assignments still build.
3. **Backends over the grammar**: ConfigSpace, Optuna, irace and racing; the space hash replaces
   the persistent-study suffix.
4. **Tree and flat encodings** for meta-optimization, after their own design (section 6).
5. **Engine items** of section 9, each as its own change with references and tests.

Tuning spaces are Experimental, so their parameters can change in a minor release; each change
says in the changelog that seeded tuning runs draw different trials, and stored assignments keep
building the configuration they built. Stable configuration fields do not change before 2.0.

## 11. Questions for review

1. Should the grammars ship as YAML resources (the source of truth) or be emitted by the
   Python builders (code as the source, grammar as an artifact)?
2. Is `pop_size` part of every grammar, or a run setting outside it, as Studio treats it today?
3. Should VAMOS read Evolver's YAML spaces directly, through the name table?
4. ConfigSpace and irace support forbidden combinations; schema 1.0.0 leaves them out until a
   space needs one. Is that acceptable?
5. A tuning `range` limits what tuners search. Is a wider `limits` field needed for manual editing
   in Studio?
6. For M4, where a configuration's automatic choice (`initializer`, `repair`) runs differently
   from every explicit value on some encoding, should the grammar offer `auto` as a value, as
   Evolver's `createInitialSolutions: default`?
