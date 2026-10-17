"""T01: generated from examples/tutorials/fundamentals.py; VAMOS 1.0.1, core dependencies only."""


def run_t01(max_evaluations=400):
    from vamos import optimize

    result = optimize(
        "zdt1",
        algorithm="nsgaii",
        n_var=30,
        pop_size=40,
        max_evaluations=max_evaluations,
        engine="numpy",
        seed=42,
    )
    front_F, indices = result.front(return_indices=True)
    front_X = result.X[indices]
    print(f"Returned decisions: {result.X.shape}")
    print(f"Returned objectives: {result.F.shape}")
    print(f"Non-dominated objectives: {front_F.shape}")
    print(f"Matching decisions: {front_X.shape}")
    print(f"Evaluations: {result.data['evaluations']}")
    return {"result": result, "front_F": front_F, "front_indices": indices}


if __name__ == "__main__":
    run_t01()
