# VAMOS-CONFIG design: parameter spaces as grammars

**Development after 1.3.0:** this design was reviewed on 2026-11-15, and the decisions of the
review are part of it (section 3). Its first three phases (section 10) ship in the version after
1.3.0; they do not describe the behaviour of any released version. The tuning spaces it discusses
are Experimental (see [Stability and versioning](../project/stability-and-versioning.md)).

VAMOS-Studio's design note
[Parameter spaces as grammars](https://github.com/vamos-optimization/VAMOS-Studio/blob/main/docs/design/parameter-spaces.md)
prototypes the model below on VAMOS 1.1.0's spaces. This design takes its direction, applies it
to the spaces of `main`, and adds what VAMOS itself has to change.

## 1. Goal

A parameter space says how an algorithm can be configured. Up to VAMOS 1.3.0 each built-in space
is a Python builder (`src/vamos/engine/tuning/racing/bridge_spaces_*.py`), and every consumer reads
it in its own way: the racing tuner and the model-based backends (`_model_backend_utils.py`,
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

## 3. Decisions

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
   the grammar into its `XConfig`; `config_from_assignment` plays that role. VAMOS keeps its own
   spaces, even where they coincide with Evolver's, and does not read Evolver's spaces directly;
   the equivalence table of section 8 exchanges configurations.
4. **The population is a run setting.** `pop_size` is not part of any grammar: the run fixes it,
   as Evolver and Studio do, and the spaces stop sampling it. A space does not know the problem
   either, so the settings that depend on it are relative and resolved when the configuration is
   built: `mutation_prob_factor` gives `factor / n_var`, and MOEA/D, NSGA-III and RVEA derive their
   population from the fixed `pop_size` and `n_obj`, with the weight vectors or reference
   directions, as they derived it from the sampled one.
5. **The grammar is data, in YAML.** Each built-in space is a YAML file under a versioned schema,
   `vamos.parameter-space` 1.0.0, packaged with VAMOS as a resource; the files are the source of
   truth. They were written once from the Python builders of VAMOS 1.3.0 and are maintained as
   YAML: no Python builder generates them. Users can load reduced or extended spaces, in YAML or
   JSON, which the same validator checks.
6. **Views are derived, never the reverse.** ConfigSpace, Optuna, irace, the tree and flat
   encodings, the CLI's tuning output and Studio's forms come from the grammar without loss.
7. **Only valid configurations.** A grammar offers only what its algorithm and encoding run, so
   every derivation is a configuration that VAMOS can build and run. The schema has no forbidden
   clauses.
8. **One range per parameter**, the same for tuners and for editing a configuration by hand in
   Studio. The schema has no second range (`limits`).
9. **`auto` is a value.** `initializer` and `repair` offer `auto`, the configuration's automatic
   choice, as Evolver's `createInitialSolutions` offers `default`, so a grammar's default can be
   the configuration's default even where no explicit value runs as it on every encoding.

## 4. The grammar document

The document has a top level (`schema`, `schema_version`, `name`, `algorithm`, `encoding`,
`description`, and the ordered `parameters` that hang from the root) and parameter nodes:

- every node: `type` (`categorical`, `real`, `int`, `boolean`), `default`, and optionally
  `description`, `label` and `role` (VAMOS's tuning roles: `structural`, `operator`,
  `operator_rate`, `population`, `adaptive`);
- numeric nodes: `range` (`[low, high]`, both included) and optionally `log`;
- categorical nodes: `values` (a list of scalars, or a mapping from each value to its production,
  which may hold `conditional_parameters`; `{}` for a value without one) and optionally
  `global_sub_parameters`, active whatever the value.

MOEA/D's real-coded variation in this form, as a sketch of M8 (section 7):

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
                crossover_alpha: {type: real, range: [0.0, 1.0], default: 0.5}
```

The validator rejects a space that breaks any of these rules, with an error that names the rule
and the place in the document:

| Rule | Statement |
| --- | --- |
| G1 | Parameter names are unique in the space and match `^[a-z][a-z0-9_]*$`. |
| G2 | Each parameter has one parent; there are no expressions. A parameter that applies to some values of a node is duplicated under each of their productions, with its own name. |
| G3 | Every parameter has a `default`, and it belongs to its domain. |
| G4 | Ranges have finite ends and are ordered, `log` needs a positive lower end, and integer ranges have integer ends. |
| G5 | `values` is not empty and has no duplicates, and no two values are written alike on the command line (`1` and `"1"`); a string value is one command-line token (not empty, no whitespace) and a number is finite; a categorical with a production has string values only, since its document is a mapping whose keys are text. |
| G6 | `schema` and `schema_version` are known; `algorithm` and `encoding` exist in VAMOS, and so does the algorithm's variant for the encoding. |
| G7 | The default configuration runs as `XConfig.default()` (decision 2). |

G1 to G6 are checked whenever a space is read. G7 runs the algorithm twice, so it is checked on
demand (`check_default_configuration(space, problem)`), and the test suite checks it for every
shipped space. A document that is not shaped as the schema says (an unknown key, a `forbidden`
clause, a `limits` range, a type that is not one of the four) is rejected as well.

A space is identified in configurations and run manifests by `name`, `schema_version` and the
SHA-256 of its canonical JSON form (insertion order kept: it is the depth-first order of the
tree), without `label` and `description`, so that editing documentation does not change it. This
identity replaces the hand-bumped suffix of the persistent Optuna studies (phase 3): `vamos tune`
appends the scoring contract and the first 16 hexadecimal digits of the hash of the space it
searches, `__vamos_cli_final_population_hv_<hash>` (SMPSO: `__vamos_cli_leaders_hv_<hash>`), so a
changed space is a new study.

PyYAML is a dependency of VAMOS, which reads the packaged YAML files at run time with its safe
loader; no generated copy ships. `tools/update_parameter_spaces.py` checks every file against G1 to
G6 and writes the snapshot of what the bridges build from the grammars.

## 5. Configurations

A configuration is a derivation tree: every categorical node has a value, and under it hang its
global sub-parameters and the parameters of the chosen production. It is valid when it holds
exactly the active parameters, each inside its domain. Because the grammar is strict, the active
parameters listed by name in depth-first order determine the tree, so three serializations carry
the same information:

- **the canonical document**, `vamos.parameter-configuration` 1.0.0, with the space's identity
  and the ordered `values`;
- **the flat mapping** (the `values` alone), which Optuna, SMAC3 and irace exchange, and which
  `config_from_assignment` reads with the run settings `pop_size` and `n_obj`;
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
| VAMOS-Studio | The tree itself: the form shows the active branch and starts from the defaults, with the grammar's single range (decision 8). |

The tuners receive a flat view of each grammar, an `AlgorithmConfigSpace` and its `ParamSpace`,
in which each production is a conditional block of its parameter and a global sub-parameter
shares the block of its node, so that a parameter of the view is active exactly when it is active
in the grammar. The views keep the grammar, and the backends derive their spaces from it as the
table says, so `_model_backend_utils.py` and `_irace_space.py` parse no condition expression for a
built-in space; a hand-written `ParamSpace` keeps that translation. `ParameterSpace.without`
removes parameters with their subtrees, which the CLI uses to search without the archive controls.
SMAC3 receives the defaults in the `ConfigurationSpace` but no enqueued configuration: its initial
design is its own.

## 7. Migrating the current spaces

The grammars were written from the builders of VAMOS 1.3.0 with these changes; `pop_size` (and
RVEA's earlier `n_partitions`) left every space (decision 4).

| Id | Change | Fixes | Scope on `main` |
| --- | --- | --- | --- |
| M1 | One categorical node `external_archive: none \| unbounded \| bounded`, with `archive_prune_policy` under `bounded`, replacing `use_external_archive`, `archive_unbounded` and the expression on them. It is Evolver's `algorithmResult` and `archiveType`. | P4 | 49 spaces |
| M2 | `mutation_eta` duplicated as `polynomial_mutation_distribution_index` and `linked_polynomial_mutation_distribution_index`, as Evolver does. | P4 | 18 spaces |
| M3 | CCMO's and CMOEA-MS's crossover choice becomes a `variation` node, `crossover_and_mutation` (with `crossover` and its probability) or `differential_evolution` (with the DE parameters), as Evolver's `variation`. | P4 | 2 spaces |
| M4 | A default for every parameter: the configuration's default where it exists, `auto` for `initial_solutions` and `repair` (decision 9), 0.2 for `scatter_base_size_factor` (a choice, and the configuration's base size of 20 for 100 individuals), and, for the operator and technique parameters that have no default of their own, a value inside the range: 0.1 for `gaussian_sigma` on real variables, and the published values for the constraint-handling techniques. | P1, P2, P3 | about 40 parameters |
| M5 | Operator rates as global sub-parameters of their operator node (`crossover_probability` under `crossover`, `mutation_probability_factor` under `mutation`, a mixed segment's probability under its operator), so that a subtree crossover moves them with their operator. | Encoding quality | every space |
| M6 | One name per operator and parameter: `polynomial` everywhere (the integer spaces' `pm` too), Evolver's snake_case names for the parameters of the name table, no value renamed to another operator's name (`uniform_reset` keeps its name: `random_reset` is the integer reset mutation), and the current names accepted when reading old assignments. The operators' own parameters (`crossover_alpha`, `undx_zeta`, ...) keep VAMOS's names. | P5 | every space |
| M7 | `mutation_probability_factor` in [0.25, 3.0] instead of an absolute probability in every space whose default probability is 1/n: the real-coded, binary and integer spaces, and NSGA-II's permutation space (default 2.0, the 2/n of `NSGAIIConfig.default()`). The permutation spaces of the other algorithms and the segments of the mixed spaces keep an absolute probability in [0.01, 0.5], which applies to a whole solution or segment and whose default is a constant. | P5 | 37 spaces |
| M8 | A `variation` node in MOEA/D's real-coded space, with `differential_evolution` and `crossover_and_mutation` productions, as Evolver's `MOEADDouble`. *In the version after 1.3.0, with `de_scheme` and `de_base` since the engine has the DE schemes.* | DE schemes in section 9 | 2 spaces, with the engine items of section 9 |

Writing the grammars also brought out two defects of the spaces of VAMOS 1.3.0. The segments of
NSGA-II's mixed space sampled their absolute mutation probabilities in [0.25, 3.0], the range of a
factor; their grammar samples [0.01, 0.5], as the other mixed spaces. And the mixed variation of
MOEA/D, NSGA-III, SMS-EMOA and SPEA2 ignored the segment operators and probabilities, so every point
of their mixed spaces ran as the default; phase 2 fixes it.

The nine mixed spaces broke G7: their segment operators draw random numbers in another order than
the default mixed operators, which apply one probability to every segment, so no point of those
spaces ran as the default configuration. *In the version after 1.3.0* the mixed grammars offer the
default operators: `crossover` and `mutation` choose `mixed` (the default, with
`crossover_probability` and `mutation_probability_factor`) or `per_segment` (the segment operators
and their probabilities), and every space passes G7.

## 8. Names

`feat/evolver-parameter-names` adds a table between VAMOS's parameter names and Evolver's, in code
and in the tuning guide: VAMOS's current name, the canonical snake_case name (Evolver's name where
Evolver has the parameter, a descriptive one otherwise), and Evolver's camelCase name. With it,
the spaces use the canonical names, the current ones remain aliases, and configurations can be
exchanged with Evolver in both directions. The configuration classes keep their Stable field
names until VAMOS 2.0, so the table also maps canonical names to configuration fields.

The grammar adopts the canonical names as its vocabulary (decision 3), and the table gains the
names that the migration adds: `linked_polynomial_mutation_distribution_index`, `variation`
(Evolver's `crossoverAndMutationVariation` and `differentialEvolutionVariation`), and `auto` for
Evolver's `createInitialSolutions: default`. Two consequences:

- a configuration published with Evolver reads into VAMOS through the table and runs in VAMOS's
  own space for the algorithm, and the other way round; VAMOS does not read Evolver's YAML spaces
  (decision 3);
- VAMOS-Studio's renamed parameters (its M2 and M8) map onto canonical names, and the Studio can
  drop its local adapter.

## 9. What remains of the MOEA/D and SMS-EMOA comparison

`feat/evolver-space-ranges` covers the extensions that need no grammar (Evolver's neighbourhood
ranges, `pbi_theta` up to 200, `modified_tchebycheff`, linked polynomial mutation, the mutation
factor, `repair` and `initializer` in MOEA/D's space). These need the grammar, the engine, or both:

| Item | Proposal | Needs |
| --- | --- | --- |
| DE schemes | `de_scheme` with `rand_1_bin`, `rand_1_exp`, `rand_2_bin` and `rand_2_exp`, under M8's `differential_evolution` production; the same crossover set as NSGA-II's space under `crossover_and_mutation`, except PCX, UNDC and simplex, which need three parents while MOEA/D pairs them; SBX's index in [5, 400] as Evolver. | MOEA/D's `DifferentialCrossoverStep` implements rand/1/bin only; CMOEA-MS already has `rand_1_exp`. Rand/2 needs five parents from the neighbourhood. *In the version after 1.3.0: the engine has the four schemes and the random base, and the grammars choose them (`de_scheme`, `de_base`); the neighbourhood's range starts at 5, the most random parents that a scheme needs (rand/2 with a random base).* |
| Normalization in the aggregation | A global sub-parameter of `aggregation_function`, `normalize_objectives` (boolean) with `normalization_epsilon` in [1e-8, 25], as Evolver; candidates `augmented_tchebycheff` and `inverted_pbi` (`ipbi_theta` in [0.01, 200]). | Normalization by the ideal and nadir points in the aggregation functions and in the compiled neighbourhood kernel, which must keep its parity with NumPy. *In the version after 1.3.0, in the engine and in every MOEA/D grammar, with `inverse_weight_tchebycheff` (Evolver's `modifiedTschebyscheff`) and `atch_rho` for the augmented Tchebycheff.* |
| Order of the subproblems | `subproblem_order: random_permutation \| cyclic`, Evolver's `subProblemIdGenerator`. | A cyclic order in MOEA/D's step; the steady-state kernels take the order as input. *In the version after 1.3.0, in the engine and in every MOEA/D grammar.* |
| Operators of SMS-EMOA and MOEA/D | The crossovers and mutations that VAMOS implements and NSGA-II's space already offers (whole arithmetic, Laplace, fuzzy recombination, BLX-αβ, Lévy flight, power law), and SBX's and polynomial mutation's indices up to 400, in both spaces; the Sobol and opposition-based initializers of NSGA-II's space. | Grammars only; the operators exist. *In the version after 1.3.0, with the Sobol, Halton and opposition-based initializers (Evolver's Cauchy initialization has no VAMOS counterpart); MOEA/D's crossovers under M8's `crossover_and_mutation`.* |
| Binary MOEA/D and SMS-EMOA | The crossover probabilities of Evolver (MOEA/D [0.6, 0.9], SMS-EMOA [0, 1]); the same aggregation, archive and selection changes as the real-coded spaces. The `bitflip` factor is M7. | Grammars only. |

## 10. Plan and compatibility

1. **Schema and validator** (`vamos.parameter-space` 1.0.0, rules G1–G7), with the configuration
   document and its three serializations. No behaviour changes. *In the version after 1.3.0.*
2. **Shipped grammars**, written once from the builders of VAMOS 1.3.0 with M1–M7 applied and
   maintained as YAML, a snapshot that detects drift between the grammars and what the bridges
   build from them, and tests of G7 and of seeded draws for every space. The bridges read the
   canonical names and the new structure, and keep the current names and structure as aliases, so
   stored assignments still build what they built. *In the version after 1.3.0.*
3. **Backends over the grammar**: ConfigSpace, Optuna (with the default configuration enqueued),
   irace and racing (both with the default configuration first); the space hash replaces the
   persistent-study suffix; Studio's forms from the grammar, with its single range. *In the version
   after 1.3.0.*
4. **Tree and flat encodings** for meta-optimization, after their own design (section 6).
5. **Engine items** of section 9 and M8, each as its own change with references and tests.

Tuning spaces are Experimental, so their parameters can change in a minor release; each change
says in the changelog that seeded tuning runs draw different trials, and stored assignments keep
building the configuration they built. Phase 2 changes every seeded tuning run, because every
space loses `pop_size`, and phase 3 changes them again, because the backends sample the tree and
start from the default configuration. Stable configuration fields do not change before 2.0.

## 11. Review

The draft asked six questions; the review of 2026-11-15 answered them, and section 3 records the
answers as decisions:

| Question | Answer |
| --- | --- |
| Should the grammars ship as YAML resources (the source of truth) or be emitted by the Python builders? | YAML resources, the source of truth, written once from the builders (decision 5). |
| Is `pop_size` part of every grammar, or a run setting outside it? | A fixed run setting, not tuned; MOEA/D, NSGA-III and RVEA derive their population from it and `n_obj` (decision 4). |
| Should VAMOS read Evolver's YAML spaces directly, through the name table? | No: VAMOS keeps its own spaces, even where they coincide; the name table exchanges configurations (decision 3). |
| Schema 1.0.0 leaves out forbidden combinations. Is that acceptable? | Yes: a grammar must not be able to express a forbidden or infeasible combination (decision 7). |
| Is a wider `limits` field needed for manual editing in Studio? | No: one range for tuning and for editing (decision 8). |
| Should the grammar offer `auto` for `initializer` and `repair`? | Yes (decision 9). |
