"""T03: generated from examples/tutorials/fundamentals.py; VAMOS 1.0.1, core dependencies only."""


def process_objectives(x):
    temperature, residence_time = x
    energy = ((temperature - 60.0) / 40.0) ** 2 + 0.25 * residence_time / 10.0
    shortfall = (100.0 - temperature) / 40.0 + 2.0 / residence_time
    return [energy, shortfall]


def process_constraints(x):
    temperature, residence_time = x
    return [450.0 - temperature * residence_time]


def solve_process(max_evaluations=400):
    import numpy as np

    from vamos import OptimizationResult, make_problem, optimize

    problem = make_problem(
        process_objectives,
        n_var=2,
        n_obj=2,
        bounds=[(60.0, 100.0), (2.0, 10.0)],
        encoding="real",
        constraints=process_constraints,
        n_constraints=1,
        name="process_design_surrogate",
    )
    result = optimize(
        problem,
        algorithm="nsgaii",
        pop_size=40,
        max_evaluations=max_evaluations,
        engine="numpy",
        seed=42,
    )
    # Recheck the original requirement, then keep all arrays aligned.
    G = np.array([process_constraints(x) for x in result.X])
    feasible_rows = np.flatnonzero(np.all(G <= 0.0, axis=1))
    if len(feasible_rows) == 0:
        raise ValueError("No feasible candidates: revisit the model or budget.")
    feasible = OptimizationResult(
        {
            "X": result.X[feasible_rows],
            "F": result.F[feasible_rows],
            "G": G[feasible_rows],
        }
    )
    front_F, local_indices = feasible.front(return_indices=True)
    return {
        "result": result,
        "G": G,
        "feasible_rows": feasible_rows,
        "feasible": feasible,
        "front_F": front_F,
        "front_indices": feasible_rows[local_indices],
    }


def run_t03(weights=(0.8, 0.2), max_evaluations=400):
    import numpy as np

    data = solve_process(max_evaluations)
    front_F = data["front_F"]
    # Normalize over this feasible non-dominated candidate set only.
    low = front_F.min(axis=0)
    span = np.ptp(front_F, axis=0)
    scaled = (front_F - low) / np.where(span > 0.0, span, 1.0)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (2,) or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Use two finite, non-negative weights with a positive sum.")
    weights = weights / weights.sum()
    scores = scaled @ weights
    position = int(np.argmin(scores))
    row = int(data["front_indices"][position])
    result = data["result"]
    x, f, g = result.X[row], result.F[row], data["G"][row, 0]
    print(f"Feasible non-dominated candidates: {len(front_F)}")
    print(f"Weights (energy, shortfall): {weights.tolist()}")
    print(f"Selected original row: {row}")
    print(f"Decision (temperature, time): ({x[0]:.6f}, {x[1]:.6f})")
    print(f"Objectives (energy, shortfall): ({f[0]:.6f}, {f[1]:.6f})")
    print(f"Constraint g: {g:.6f}")
    print(f"Normalized weighted score: {scores[position]:.6f}")
    data.update({"selected_row": row, "weights": weights, "scores": scores})
    return data


if __name__ == "__main__":
    run_t03()
