#!/usr/bin/env python3
"""Minimal command-line runner for one MOUT system/proposition pair."""

from __future__ import annotations

import argparse
import json
import random
import runpy
from pathlib import Path

import numpy as np
from jmetal.algorithm.multiobjective.nsgaiii import (
    UniformReferenceDirectionFactory,
)
from jmetal.operator.selection import BinaryTournamentSelection
from jmetal.util.termination_criterion import StoppingByEvaluations
from pymoo.indicators.hv import HV

from mout import (
    ConfigDatasetProblem,
    ConfigurationCrossover,
    ConfigurationMutation,
    MOUT,
    ReferenceVector,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MOUT on one system and one proposition."
    )
    parser.add_argument("--system", default="HSQLDB")
    parser.add_argument(
        "--proposition",
        default="data1",
        help="Variable name in dataset/propositions/Data_<system>.txt.",
    )
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--population-size", type=int, default=10)
    parser.add_argument("--partitions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: MOUT3/results/<system>/<proposition>.",
    )
    return parser.parse_args()


def load_proposition(path: Path, name: str):
    namespace = runpy.run_path(str(path))
    if name not in namespace:
        available = sorted(
            key for key, value in namespace.items()
            if key.startswith("data") and isinstance(value, list)
        )
        raise KeyError(
            f"{name!r} is not defined in {path}. Available: {available}"
        )
    proposition = namespace[name]
    if not isinstance(proposition, list) or not proposition:
        raise ValueError(f"{name!r} in {path} is not a non-empty proposition")
    return proposition


def main() -> None:
    args = parse_args()
    if args.budget <= 0 or args.population_size <= 1:
        raise ValueError("budget must be positive and population-size must exceed 1")

    project_root = Path(__file__).resolve().parents[1]
    system_csv = project_root / "dataset" / "systems" / f"{args.system}.csv"
    proposition_file = (
        project_root
        / "dataset"
        / "propositions"
        / f"Data_{args.system}.txt"
    )
    if not system_csv.is_file():
        raise FileNotFoundError(f"System dataset not found: {system_csv}")
    if not proposition_file.is_file():
        raise FileNotFoundError(f"Proposition file not found: {proposition_file}")

    proposition = load_proposition(proposition_file, args.proposition)
    objective_count = len(proposition)

    random.seed(args.seed)
    np.random.seed(args.seed)

    problem = ConfigDatasetProblem(
        str(system_csv),
        data=proposition,
        obj_num=objective_count,
        new_data=None,
    )
    directions = UniformReferenceDirectionFactory(
        n_dim=objective_count,
        n_partitions=args.partitions,
    )
    problem.reference_vectors = [
        ReferenceVector(direction) for direction in directions.compute()
    ]

    algorithm = MOUT(
        problem=problem,
        population_size=args.population_size,
        reference_directions=directions,
        crossover=ConfigurationCrossover(0.9),
        mutation=ConfigurationMutation(
            problem,
            probability=1.0 / max(1, problem.number_of_variables()),
        ),
        selection=BinaryTournamentSelection(),
        termination_criterion=StoppingByEvaluations(
            max_evaluations=args.budget
        ),
        budget=args.budget,
    )

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else project_root / "results" / args.system / args.proposition
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    algorithm.run(
        seed=args.seed,
        progress_root=str(output_dir),
        exp_name=f"seed_{args.seed}",
    )

    front = np.asarray(
        [solution.solution_ori_objectives for solution in algorithm.solutions],
        dtype=float,
    )
    finite_front = front[np.isfinite(front).all(axis=1)]
    hv = (
        float(HV(ref_point=np.full(objective_count, 0.1))(-finite_front))
        if finite_front.size
        else 0.0
    )
    result = {
        "system": args.system,
        "proposition": args.proposition,
        "seed": args.seed,
        "budget": args.budget,
        "evaluations": int(algorithm.evaluations),
        "hv_reference_point": [0.1] * objective_count,
        "hv": hv,
        "front": finite_front.tolist(),
    }
    result_path = output_dir / f"seed_{args.seed}.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
