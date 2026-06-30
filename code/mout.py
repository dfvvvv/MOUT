# -*- coding: utf-8 -*-
"""Standalone core MOUT implementation."""
from __future__ import annotations

import copy
import json
import math
import os
import random

import numpy as np
import pandas as pd
from jmetal.algorithm.multiobjective.nsgaiii import NSGAIII
from jmetal.core.problem import IntegerProblem
from jmetal.core.solution import IntegerSolution
from jmetal.operator.crossover import SPXCrossover
from jmetal.operator.mutation import BitFlipMutation
from pymoo.indicators.hv import HV

class ReferenceVector:
    """
    Represents a reference direction vector in the objective space,
    with reward accumulation and decay.

    Attributes:
        coords (np.ndarray): Coordinates of the reference vector.
        reward_value (float): Current accumulated reward.
    """
    def __init__(self, coords):
        self.coords = np.array(coords, dtype=float)
        self.reward_value = 0.0
        self.other_reward = 0.0
        self.association = 0

    def get_reward(self, all_vectors, k=3):
        """
        Increment this vector's reward by 1, then distribute 1/10 of that reward
        to the k nearest other reference vectors.

        Args:
            all_vectors (List[ReferenceVector]): List of all reference vectors.
            k (int): Number of nearest neighbors to reward.
        """
        increment = 1.0
        self.reward_value += increment

        # Compute distances to other vectors
        distances = []
        for vec in all_vectors:
            if vec is not self:
                dist = np.linalg.norm(vec.coords - self.coords)
                distances.append((dist, vec))

        # Find k nearest neighbors
        distances.sort(key=lambda x: x[0])
        nearest = [vec for _, vec in distances[:k]]

        # Distribute 1/10 of the increment to each neighbor
        neighbor_reward = increment * 0.1
        for neighbor in nearest:
            neighbor.reward_value += neighbor_reward

    def decay_reward(self, decay_rate=0.25):
        """
        Apply decay to the reward value, reducing it by the given rate.

        Args:
            decay_rate (float): Fraction to decay each generation (default 0.25).
        """
        self.reward_value *= (1.0 - decay_rate)

    def distance_to(self, point):
        """MOUT implementation detail."""
        f = -np.array(point, dtype=float)
        v = self.coords
        proj = (f.dot(v) / v.dot(v)) * v
        return np.linalg.norm(f - proj)

    def __repr__(self):
        return f"ReferenceVector(coords={self.coords.tolist()}, reward={self.reward_value:.4f})"

class Fragment:
    """
    A segment of a piecewise function defined by performance thresholds.

    Attributes:
        left (float): Lower bound of performance domain.
        right (float): Upper bound of performance domain.
        down (float): Score at lower bound.
        up (float): Score at upper bound.
    """
    def __init__(self, left, right, down, up):
        self.left = left
        self.right = right
        self.down = down
        self.up = up

    def contains(self, performance):
        """Check if the performance lies within this fragment's domain."""
        return self.left <= performance <= self.right

    def compute_score(self, performance) -> float:
        """
        Compute the fragment score based on a linear mapping.

        Args:
            performance (float): The performance value.

        Returns:
            float: Mapped score.
        """
        if self.left == -float("inf"):
            return 0
        elif self.right == float("inf"):
            return 1
        else:
            return self.down + (performance - self.left) / (self.right - self.left) * (self.up - self.down)

class Proposition:
    """
    A piecewise function composed of multiple Fragments.

    Attributes:
        fragments (List[Fragment]): Components of the piecewise function.
    """
    def __init__(self, fragments:list[Fragment]):
        self.fragments = fragments
        self.num_front = 0
        self.num_niching = 0

    def compute_sat_score(self, performance):
        """
        Compute the satisfaction score by finding the fragment containing the performance
        and evaluating it. If no fragment matches, returns 0.

        Args:
            performance (float): Performance value for this proposition.

        Returns:
            float: The fragment evaluation result.
        """
        for frag in self.fragments:
            if frag.contains(performance):
                return frag.compute_score(performance)
        return 0.0

    def mutate(self, gap: float) -> None:
        """MOUT implementation detail."""
        import random
        if len(self.fragments) < 2:
            raise ValueError("Proposition must contain at least 2 fragments to perform mutation.")
        if random.random() <= 0.8:
            idx = random.randint(0, len(self.fragments) - 2)
            if idx == 0:
                left_point = self.fragments[idx].right - gap / 2
                right_point = (self.fragments[idx + 1].right + self.fragments[idx].right) / 2
            elif idx == len(self.fragments) - 2:
                left_point = (self.fragments[idx].left + self.fragments[idx].right) / 2
                right_point = self.fragments[idx].right + gap / 2
            else:
                left_point = self.fragments[idx].left
                right_point = self.fragments[idx + 1].right
            down = self.fragments[idx].down
            up = self.fragments[idx + 1].up
            r = random.random()
            if r <= 0.1:
                tmp = down
            elif r >= 0.9:
                tmp = up
            else:
                tmp = random.uniform(down, up)
            self.fragments[idx].up = tmp
            self.fragments[idx + 1].down = tmp
            split = random.uniform(left_point, right_point)
            self.fragments[idx].right = split
            self.fragments[idx + 1].left = split

class Propositions:
    """
    A collection of Proposition instances and associated reference vectors.

    Attributes:
        props (List[Proposition]): The proposition functions.
        reference_vectors (List[ReferenceVector]): Reference directions for association.
    """
    def __init__(self, props:list[Proposition]):
        self.props = props
        self.fitness = 0

class NewIntegerSolution(IntegerSolution):
    """MOUT implementation detail."""
    def __init__(self, lower_bound, upper_bound, n_objectives):
        super().__init__(lower_bound, upper_bound, n_objectives)
        self.solution_ori_objectives = [0.0] * n_objectives
        self.solution_ori_value = [0.0] * n_objectives

class ConfigDatasetProblem(IntegerProblem):
    """Core implementation used by the MOUT run loop."""

    def __init__(self, csv_path: str, data, obj_num, new_data):
        super().__init__()
        self.df = pd.read_csv(csv_path, sep=r'[;,]', engine='python')
        # if 'revision' in self.df.columns:
        #     self.df = self.df.drop(columns=['revision'])
        self.feature_cols   = self.df.columns[:-obj_num].tolist()  #
        self.objective_cols = self.df.columns[-obj_num:].tolist()  #

        self.lower_bound = [0] * self.number_of_variables()
        self.upper_bound = [1] * self.number_of_variables()

        self.obj_directions = [self.MAXIMIZE] * self.number_of_objectives()
        self.obj_labels     = self.objective_cols
        self.data = data  #
        self.new_data = new_data
        self.initial_propositions = None
        self.variables_propositions = None
        self.reference_vectors = None
        self.independent_set = []

    def number_of_variables(self) -> int:
        return len(self.feature_cols)

    def number_of_objectives(self) -> int:
        return len(self.objective_cols)

    def number_of_constraints(self) -> int:
        return 0

    def name(self) -> str:
        return "ConfigDatasetProblem"

    def evaluate(self, solution: NewIntegerSolution,num=0) -> NewIntegerSolution:
        mask = (self.df[self.feature_cols] == solution.variables).all(axis=1)
        if mask.sum() == 1:
            row = self.df.loc[mask, self.objective_cols].iloc[0]
            for i, col in enumerate(self.objective_cols):
                raw_value = float(row[col])
                solution.solution_ori_value[i] = raw_value
                prop = self.variables_propositions.props[i]
                sat = prop.compute_sat_score(raw_value)
                ori_sat = self.initial_propositions.props[i].compute_sat_score(raw_value)
                solution.solution_ori_objectives[i] = ori_sat
                solution.objectives[i] = sat
        else:
            solution.objectives = [-float("inf")] * self.number_of_objectives()
            solution.solution_ori_value = [-float("inf")] * self.number_of_objectives()
            solution.solution_ori_objectives = [-float("inf")] * self.number_of_objectives()

        return solution

    def search_ref_vector(self, solution: NewIntegerSolution) -> ReferenceVector:
        """MOUT implementation detail."""
        objectives = solution.objectives

        distances = [rv.distance_to(objectives) for rv in self.reference_vectors]

        min_index = np.argmin(distances)
        return self.reference_vectors[min_index]

    def create_solution(self) -> NewIntegerSolution:
        if self.independent_set == []:
            for column in self.df.columns[:self.number_of_variables()]:
                self.independent_set.append(self.df[column].unique().tolist())
        row = self.df.sample(n=1).iloc[0]

        solution = NewIntegerSolution(
            self.lower_bound,
            self.upper_bound,
            self.number_of_objectives()
        )
        for i, col in enumerate(self.feature_cols):
            solution.variables[i] = float(row[col])

        solution = self.evaluate(solution)
        print("create solution:", solution.variables, "->", solution.objectives)
        return solution

    def create_propositions(self) -> Propositions:
        proposition_list = []
        for prop_data in self.data:
            fragment_list = []
            for left, right, down, up in prop_data:
                if left == "-infinity":
                    left = -float('inf')
                elif right == "infinity":
                    right = float('inf')
                current_fragment = Fragment(left, right, down, up)
                fragment_list.append(current_fragment)
            proposition = Proposition(fragment_list)
            proposition_list.append(proposition)
        self.initial_propositions = Propositions(proposition_list)
        proposition_list2 = []
        print("new_data",self.new_data)
        if self.new_data is not None:
            for prop_data in self.new_data:
                fragment_list = []
                for left, right, down, up in prop_data:
                    if left == "-infinity":
                        left = -float('inf')
                    elif right == "infinity":
                        right = float('inf')
                    current_fragment = Fragment(left, right, down, up)
                    fragment_list.append(current_fragment)
                proposition = Proposition(fragment_list)
                proposition_list2.append(proposition)
            self.variables_propositions = Propositions(proposition_list2)
        else:
            self.variables_propositions = copy.deepcopy(self.initial_propositions)

class MOUT(NSGAIII):
    """Core implementation used by the MOUT run loop."""

    # RQ4 defaults. Tested pairs: (1, 5), (1, 10), (3, 8), and (5, 10).
    SUSTAINABLE_THRESHOLD = 1
    TEMPORARY_THRESHOLD = 5

    def __init__(self, *args, max_stagnation: int = 100, max_try_limit: int = 300, budget: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_try = 0
        self.budget = int(budget if budget is not None else 300)
        self.max_try_limit = max_try_limit
        self.offspring_count = self.population_size
        self.stagnation_count = 0
        self.max_stagnation = max_stagnation  #
        self.prev_igd = None
        self.pf_x = None
        self.pf_y = None
        self._evaluated_keys = set()
        self.num_front = 0
        self.num_niching = 0
        self.count = 0
        self.system_name = None
        self.cnt = None
        self.archive = []
        self.last_gen = 0
        self.last_budget = 0
        self.current_mode = "initial"

    def repair_variable_propositions(self,
                                     step_frac: float = 0.05,
                                     min_width: float = 1e-6,
                                     threshold_ratio: float = 0.5):
        """MOUT implementation detail."""
        P = self.problem.variables_propositions
        M = len(P.props)
        pop = self.solutions
        N = len(pop)

        def nonall1_count():
            """MOUT implementation detail."""
            cnt = 0
            for sol in pop:
                scores = [
                    P.props[i].compute_sat_score(sol.solution_ori_value[i])
                    for i in range(M)
                ]
                if any(s < 1.0 for s in scores):
                    cnt += 1
            return cnt

        if nonall1_count() >= N:
            width_list = []
            for prop in P.props:
                frags = prop.fragments
                pen = frags[-2]
                width_list.append(pen.right - pen.left)
            avg_width = sum(width_list)/len(width_list)

            while nonall1_count() < threshold_ratio * N:
                delta = max(avg_width * step_frac, min_width)
                for prop in P.props:
                    frags = prop.fragments
                    if len(frags) < 3:
                        continue
                    pen = frags[-2]
                    new_cut = pen.right + delta
                    pen.right      = new_cut
                    frags[-1].left = new_cut
                step_frac *= 0.5
                if avg_width * step_frac < min_width:
                    break

    def replacement(self, population: list[NewIntegerSolution], offspring_population: list[NewIntegerSolution]):
        all_solutions = population + offspring_population
        for sol in all_solutions:
            sol.objectives = [-o for o in sol.objectives]

        replacement_result = super().replacement(population, offspring_population)
        survivors = (
            replacement_result[0]
            if isinstance(replacement_result, tuple)
            else replacement_result
        )

        for sol in all_solutions:
            sol.objectives = [-o for o in sol.objectives]

        return survivors

    def get_offspring(self, population):
        offspring = []
        while len(offspring) < self.offspring_count:
            #    / Perform one crossover on the population, yielding 2 children
            # parent1, parent2 = random.sample(population, 2)

            parent1 = self.selection_operator.execute(population)
            parent2 = self.selection_operator.execute(population)

            children = self.crossover_operator.execute([parent1, parent2])

            #    / Then mutate each child and collect them
            for child in children:
                mutated = self.mutation_operator.execute(child)
                offspring.append(mutated)
                # / Stop once we've reached the target count
                if len(offspring) >= self.offspring_count:
                    break

        return offspring

    def create_initial_solutions(self) -> list[NewIntegerSolution]:
        """MOUT implementation detail."""
        # solutions = [ self.problem.create_fix_solution() for _ in range(self.population_size) ]
        solutions = [ self.problem.create_solution() for _ in range(self.population_size) ]
        self.archive = copy.deepcopy(solutions)
        print("Evaluating initial population...:", solutions)
        return solutions

    def evaluate(self, solution_list: list[NewIntegerSolution]) -> list[NewIntegerSolution]:
        """MOUT implementation detail."""
        num = 0
        evaluated = []
        for sol in solution_list:
            evaluated.append(self.problem.evaluate(sol,num))
        return evaluated

    def init_progress(self) -> None:
        """MOUT implementation detail."""
        self.evaluations = 0
        self.max_try = 0

    def stopping_condition_is_met(self) -> bool:
        """MOUT implementation detail."""
        if self.max_try > self.max_try_limit:
            return True
        return self.termination_criterion.is_met

    def step(self) -> None:
        """MOUT implementation detail."""
        offspring = self.get_offspring(self.solutions)
        evaluated = []

        for child in offspring:
            child = self.problem.evaluate(child)
            if any(obj == -float('inf') for obj in child.solution_ori_value):
                self.max_try += 1
            evaluated.append(child)
            if self.max_try > getattr(self, 'max_try_limit', float('inf')):
                break

        filtered = []
        for child in evaluated:
            same_count = sum(1 for sol in self.solutions if sol.variables == child.variables)
            if same_count < 1:
                filtered.append(child)
                if not any(obj == -float('inf') for obj in child.solution_ori_value):
                    key = tuple(child.variables)
                    if key not in self._evaluated_keys:
                        self.evaluations += 1
                        self._evaluated_keys.add(key)
        evaluated = filtered

        self.tmp_solution = copy.deepcopy(self.solutions)
        self.tmp_offspring = copy.deepcopy(evaluated)
        self.solutions = self.replacement(self.solutions, evaluated)

    def create_initial_propositions(self) -> Propositions:
        self.problem.create_propositions()

    def run(self, seed, cnt = None, progress_root: str = "progress", exp_name: str = "run") -> None:
        """MOUT implementation detail."""
        self.init_progress()
        self.create_initial_propositions()

        print("Creating initial set of solutions...")
        self.solutions = self.create_initial_solutions()

        print("Evaluating solutions...")
        self.solutions = self.evaluate(self.solutions)
        print("Running main loop until termination criteria is met")
        self.problem.initial_propositions.fitness = self.update_ref_reward()
        pre_hv = 0
        while not (
            any(all(v == 1.0 for v in sol.solution_ori_objectives) for sol in self.solutions)
            or self.evaluations > self.budget
            or self.stopping_condition_is_met()
        ):
            self.max_try = 0
            print("generation:", self.termination_criterion.evaluations)
            print("evaluations:", self.evaluations)
            
            if all(all(v == 1.0 for v in sol.objectives) for sol in self.solutions):
                print("All solutions have original objectives equal to 1.0, repairing propositions...")
                self.repair_variable_propositions()
                self.solutions = self.evaluate(self.solutions)
            sys_name = os.path.basename(progress_root)
            pre_solutions = copy.deepcopy(self.solutions)
            self.system_name = sys_name
            self.cnt = cnt
            self.seed = seed
            self.step()
            hv, improved = self.compute_Hypervolume()
            if hv is not None and pre_hv is not None and hv < pre_hv:
                print("Hypervolume decreased from {:.6f} to {:.6f}, reverting to previous solutions.".format(pre_hv, hv))
                self.solutions = copy.deepcopy(pre_solutions)
            else:
                pre_hv = hv
            self.update_archive()
            # risk, info = self.local_optimum_risk()
            # print("risk:", risk)
            cand = None
            # RQ3/RQ4 EXPERIMENT BLOCK START
            if self.stagnation_count <= self.SUSTAINABLE_THRESHOLD:
                print("0")
                mode = "improving"
                self.current_mode = mode
                self.change_propositions(mode)
            elif self.stagnation_count <= self.TEMPORARY_THRESHOLD:
                mode = "temp"
                self.current_mode = mode
                self.change_propositions(mode)
            else:
                mode = "stagnant"
                self.current_mode = mode
                self.change_propositions(mode)
            # RQ3/RQ4 EXPERIMENT BLOCK END
            
            file = os.path.join(os.path.dirname(__file__), "log", "2.6/archiveNoback", sys_name, str(cnt), str(seed))
            # record(self.termination_criterion.evaluations, hv, improved, mode, file)
            self.solutions = self.evaluate(self.solutions)
            # if self.stagnation_count > self.max_stagnation:
            #     print("info:", info)
            #     self.change_propositions()
            #     cur_prop = self.problem.variables_propositions
            #     init_prop = self.problem.initial_propositions
            #     # print("same" if (cur_prop and init_prop and
            #     #                 [[(f.left, f.right, f.down, f.up) for f in pr.fragments] for pr in cur_prop.props] ==
            #     #                 [[(f.left, f.right, f.down, f.up) for f in pr.fragments] for pr in init_prop.props]) else "not")
            #     # self.change_propositions_random()
            #     # self.change_propositions_nd()
            #     if cnt != None:
            #         self.save_variable_propositions_txt(sys=sys_name, cnt=cnt, seed=seed)
            #     self.solutions = self.evaluate(self.solutions)
            # self.compute_Hypervolume()
            # if self._should_change_proposition():
            #     self._switch_proposition_policy(sys_name=sys_name, cnt=cnt, seed=seed)
            self.termination_criterion.update()
            self._maybe_log_progress(progress_root, exp_name)
        self._maybe_log_progress(progress_root, exp_name, True)
        self.update_archive()                      #
        self.finalize_solutions_from_archive(k=10)
        # self.solutions = self.evaluate(self.solutions)

        if any(all(v == 1.0 for v in sol.solution_ori_objectives) for sol in self.solutions):
            print("All original objectives are 1.0, terminating early.")
        elif self.evaluations > self.termination_criterion.max_evaluations:
            print("Evaluation count exceeded 100, terminating.")
        else:
            print("Finished by stopping condition.")

    def compute_Hypervolume(self):
        """MOUT implementation detail."""
        front = -np.array([sol.solution_ori_objectives for sol in self.solutions])
        
        ref_point = np.ones(front.shape[1]) * 0.1
        print(ref_point)

        hv_indicator = HV(ref_point=ref_point)
        hv_value = hv_indicator(front)
        print(hv_value)

        if not hasattr(self, 'prev_hv'):
            self.prev_hv = hv_value
            improved = False
        else:
            improved = hv_value > self.prev_hv
            if not improved:
                self.stagnation_count += 1
                print("No improvement in Hypervolume. Stagnation count:", self.stagnation_count)
            else:
                self.archive = copy.deepcopy(self.solutions)
                self.count += 1
                self.max_try = 0
                self.stagnation_count = 0
                print("improvement in Hypervolume. Stagnation count:", self.stagnation_count)
            self.prev_hv = hv_value

        return hv_value, improved

    def generate_propositions(self,
                          low: float,
                          max_val: float,
                          num_frags: int
                          ) -> Proposition:
        """MOUT implementation detail."""
        fragments: list[Fragment] = []
        gap = (max_val - low) / num_frags
        first_right = random.uniform(low, low + gap)
        fragments.append(Fragment(-float('inf'), first_right, 0.0, 0.0))

        prev_up = 0.0
        left = first_right

        for i in range(num_frags - 2):
            right = left + random.uniform(gap / 2, gap)
            down = prev_up
            if random.random() < 0.5:
                up = down
            else:
                up = down + random.uniform(0.1, 0.25)
            up = min(up, 1.0)
            fragments.append(Fragment(left, right, down, up))

            left = right
            prev_up = up

        if left > max_val:
            left = max_val
        
        fragments.append(Fragment(left, max_val, prev_up, 1.0))
        fragments.append(Fragment(max_val, float('inf'), 1.0, 1.0))

        return Proposition(fragments)

    def evaluate_propositions(self, p: Propositions, mode: str):
        """MOUT implementation detail."""
        import numpy as np
        from pymoo.indicators.hv import HV
        from jmetal.util.ranking import FastNonDominatedRanking
        from jmetal.util.comparator import DominanceComparator
        parent = list(getattr(self, "tmp_solution", []) or []) + list(getattr(self, "tmp_offspring", []) or [])
        parent2 = copy.deepcopy(parent)
        if not parent:
            p.fitness = -1e9
            return

        M = int(self.problem.number_of_objectives())

        obj_bak = [(s, list(s.objectives)) for s in parent]
        ip_bak  = None if getattr(self, "ideal_point", None) is None else self.ideal_point.copy()
        wp_bak  = None if getattr(self, "worst_point", None) is None else self.worst_point.copy()
        try:
            ep_bak = None if getattr(self, "extreme_points", None) is None else [e.copy() for e in self.extreme_points]
        except Exception:
            ep_bak = getattr(self, "extreme_points", None)
        rng_bak = np.random.get_state()

        try:
            for s in parent:
                s.objectives = [p.props[i].compute_sat_score(s.solution_ori_value[i]) for i in range(M)]
            for s in parent2:
                s.objectives = [self.problem.initial_propositions.props[i].compute_sat_score(s.solution_ori_value[i]) for i in range(M)]
            ranking = FastNonDominatedRanking(DominanceComparator())
            ranking.compute_ranking(parent, k=self.population_size)
            fronts = ranking.ranked_sublists
            remaining = self.population_size
            front_sizes = []
            for front in fronts:
                front_len = len(front)
                
                if front_len <= remaining:
                    front_sizes.append(front_len)
                    remaining -= front_len
                else:
                    front_sizes.append(front_len)
                    break  #
            ranking2 = FastNonDominatedRanking(DominanceComparator())
            ranking2.compute_ranking(parent2, k=self.population_size)
            fronts2 = ranking2.ranked_sublists
            remaining = self.population_size
            front_sizes2 = []
            for i, front in enumerate(fronts2):
                front_len = len(front)
                
                if front_len <= remaining:
                    front_sizes2.append(front_len)
                    remaining -= front_len
                else:
                    front_sizes2.append(front_len)
                    remaining = 0
                    break  #
            last = len(front_sizes2)
            
            def evaluate_proposition_fitness(front1, front2):
                """MOUT implementation detail."""
                length_now = len(front1)
                length_base = len(front2)
                N = self.population_size
                delta = 0.03  #

                # ---------- front1 ----------
                pre1 = front1[:-1]
                L1 = front1[-1]
                A1 = sum(pre1)
                R1 = N - A1

                if length_now == 1:
                    return -1e9
                if L1 <= 0:
                    return -1e9
                if R1 <= 0:
                    exploit_total1 = 1.0 * (length_now / length_base)
                    explore1 = 0.0
                    p_keep1 = 1.0
                else:
                    if R1 > L1:
                        R1 = L1
                    p_keep1 = R1 / L1
                    explore1 = (1.0 - p_keep1) * (R1 / N)
                    exploit_total1 = ((A1 / N) + (R1 / N) * p_keep1) * (length_now / length_base)

                # ---------- front2 ----------
                pre2 = front2[:-1]
                L2 = front2[-1]
                A2 = sum(pre2)
                R2 = N - A2

                if L2 <= 0:
                    return -1e9
                if R2 <= 0:
                    exploit_total2 = 1.0
                    explore2 = 0.0
                    p_keep2 = 1.0
                else:
                    if R2 > L2:
                        R2 = L2
                    p_keep2 = R2 / L2
                    exploit_total2 = (A2 / N) + (R2 / N) * p_keep2
                    explore2 = (1.0 - p_keep2) * (R2 / N)

                if mode == "improving":
                    # if pre1 != pre2:
                    #     return -1e9
                    # target = explore2 + delta
                    # if target > 0.5:
                    #     target = 0.5
                    return (explore1 - explore2)

                elif mode == "temp":
                    # if pre1 != pre2:
                    #     return -1e9
                    # target = explore2 - delta
                    # if target < 0.0:
                    #     target = 0.0
                    return (exploit_total1 - exploit_total2)

                else:  # stag
                    # uncertain_count1 = max(0, L1 - R1)
                    return (explore1 - explore2)
                    
                # if mode == "improving":
                #     now = (10-sum(front1[:-1]))/front1[-1]
                #     if now < base:
                #         return -100
                #     return base - now
                # elif mode == "temp":
                #     now = (10-sum(front1[:-1]))/front1[-1]
                #     if now > base:
                #         return -100
                #     return base - now
                # else:
                #     now = (10-sum(front1[:-1]))/front1[-1]
                #     if now < base:
                #         return -100
                #     return now - base
                # if(len(front1) != len(front2)):
                #     return 0
                # for i in range(last-1):
                #     if front2[i]!=front1[i]:
                #         return 0
                # return front1[-1]
            
            p.fitness = evaluate_proposition_fitness(front_sizes, front_sizes2)
            print("front size ",front_sizes, " fitness ", p.fitness, " mode ", mode)
            # hv_base = 0.0
            # try:
            #     front_pool_raw = np.array(
            #         [s.solution_ori_objectives for s in self.solutions],
            #         dtype=float
            #     )
            #     front_pool_min = -front_pool_raw
            #     hv_base = float(HV(ref_point=[0.1] * M)(front_pool_min))
            # except Exception as e:
            #     print("[EvalPA] failed to compute hv_base:", e)
            #     hv_base = 0.0

            # K_RUNS = 10
            # surv_sets = []

            # parent = copy.deepcopy(list(getattr(self, "tmp_solution", []) or []))
            # offspring = copy.deepcopy(list(getattr(self, "tmp_offspring", []) or []))

            # for k in range(K_RUNS):
            #     surv_k , _ , _ = self.replacement(parent, offspring)
            #     if not surv_k:
            #         p.fitness = 0.0
            #         return

            #     try:
            #         front_new_raw = np.array(
            #             [s.objectives for s in surv_k],
            #             dtype=float
            #         )
            #         front_new_min = -front_new_raw
            #         hv_new = float(HV(ref_point=[0.1] * M)(front_new_min))
            #     except Exception as e:
            #         print("[EvalPA] failed to compute hv_new:", e)
            #         p.fitness = 0.0
            #         return

            #         print("[EvalPA] HV dropped in shadow run, rejecting pa.")
            #         p.fitness = 0.0
            #         return

            #     surv_sets.append(ids_k)

            # # surv_k, num_front, num_niching = self.replacement(parent, offspring)

            # # if not surv_k:
            # #     p.fitness = 0.0
            # #     return

            # # try:
            # #     front_new_raw = np.array(
            # #         [s.solution_ori_objectives for s in surv_k],
            # #         dtype=float
            # #     )
            # #     front_new_min = -front_new_raw
            # #     hv_new = float(HV(ref_point=[0.1] * M)(front_new_min))
            # # except Exception as e:
            # #     print("[EvalPA] failed to compute hv_new:", e)
            # #     p.fitness = 0.0
            # #     return

            # #     print("[EvalPA] HV dropped in shadow run, rejecting pa.")
            # #     p.fitness = 0.0
            # #     return
            
            # pressure_score = 1.0
            # if len(surv_sets) > 1:
            #     sims = []
            #     for i in range(len(surv_sets)):
            #         for j in range(i + 1, len(surv_sets)):
            #             A, B = surv_sets[i], surv_sets[j]
            #             inter = len(A & B)
            #             union = len(A | B)
            #             sims.append(inter / (union + 1e-12))
            #     if sims:
            #         pressure_score = float(np.mean(sims))


            # # evals = float(getattr(self, "generation", 0))
            # # max_evals = 100

            # # p.fitness = hv_new + (1.0 - lam) * pressure_score + lam * (1.0 - pressure_score)

            # # hv_diff = hv_new - hv_base
            # # front_diff = (num_front - self.num_front)
            # # niching_diff = (num_niching - self.num_niching)
            # p.fitness = 1 - pressure_score
            # print(f"[EvalPA] proposition fitness: pressure_score={pressure_score:.6f} -> fitness={p.fitness:.6f}")
            

        finally:
            for s, obj in obj_bak:
                s.objectives = obj
            self.ideal_point    = ip_bak
            self.worst_point    = wp_bak
            self.extreme_points = ep_bak
            np.random.set_state(rng_bak)

    def _dominates_max(self, A, B, eps: float = 1e-12) -> bool:
        """MOUT implementation detail."""
        return all((B[i] >= A[i] + eps) or abs(B[i] - A[i]) <= eps for i in range(len(A))) and \
            any(B[i] > A[i] + eps for i in range(len(A)))

    def update_archive(self):
        """MOUT implementation detail."""
        import math, copy

        pool = (self.archive or []) + (self.solutions or [])

        uniq = {}
        for s in pool:
            F = getattr(s, "solution_ori_objectives", None)
            if not F:
                continue
            if any(not math.isfinite(float(v)) for v in F):
                continue
            key = tuple(s.variables)
            if key not in uniq:
                uniq[key] = copy.deepcopy(s)
        pool = list(uniq.values())

        nd = []
        Fs = [list(map(float, s.solution_ori_objectives)) for s in pool]
        n = len(pool)
        for i in range(n):
            dominated = False
            for j in range(n):
                if i == j:
                    continue
                if self._dominates_max(Fs[i], Fs[j]):  # j dominates i
                    dominated = True
                    break
            if not dominated:
                nd.append(pool[i])

        self.archive = nd

    def finalize_solutions_from_archive(self, k: int = 10):
        """MOUT implementation detail."""
        import math, copy

        pool = (self.archive or []) + (self.solutions or [])
        uniq = {}
        for s in pool:
            F = getattr(s, "solution_ori_objectives", None)
            if not F:
                continue
            if any(not math.isfinite(float(v)) for v in F):
                continue
            key = tuple(s.variables)
            if key not in uniq:
                uniq[key] = s
        pool = list(uniq.values())

        if not pool:
            return

        def dominates_max(a, b, eps: float = 1e-12) -> bool:
            # a dominates b (maximize)
            return all(a[i] >= b[i] - eps for i in range(len(a))) and any(a[i] > b[i] + eps for i in range(len(a)))

        def first_front(sols):
            Fs = [list(map(float, s.solution_ori_objectives)) for s in sols]
            front = []
            for i, si in enumerate(sols):
                dominated = False
                for j, sj in enumerate(sols):
                    if i == j:
                        continue
                    if dominates_max(Fs[j], Fs[i]):  # sj dominates si
                        dominated = True
                        break
                if not dominated:
                    front.append(si)
            return front

        selected = []
        remaining = pool[:]
        while remaining and len(selected) < k:
            f1 = first_front(remaining)

            f1_keys = {tuple(s.variables) for s in f1}
            remaining = [s for s in remaining if tuple(s.variables) not in f1_keys]

            need = k - len(selected)
            if len(f1) <= need:
                selected.extend(f1)
            else:
                f1_sorted = sorted(f1, key=lambda s: float(sum(s.solution_ori_objectives)), reverse=True)
                selected.extend(f1_sorted[:need])
                break

        self.solutions = copy.deepcopy(selected)
        self.solutions = self.evaluate(self.solutions)

    def change_propositions(self,mode:str):
        #self.stag_update_ref_reward()
        num_props = 100
        gen = 0
        max_gen = 100
        new_props = []
        gap = 0
        # with open(log_path, "a", encoding="utf-8") as f:
        #     f.write(f"New change_prop\n")
        for i in range(num_props):
            props = []
            for i in range(len(self.problem.initial_propositions.props)):
                frags = self.problem.initial_propositions.props[i].fragments
                v = sorted(div.solution_ori_value[i] for div in self.solutions)
                low = random.uniform(v[0], v[len(v)//2])
                max_val = random.uniform(v[-2], v[-1]*1.1)
                
                num_frags = len(frags)
                prop = self.generate_propositions(low, max_val, num_frags)
                prop.mutate((v[-1] - v[0])/5)
                if prop == []:
                    props = []
                    break
                props.append(prop)
            cand = Propositions(props)         #
            self.evaluate_propositions(cand,mode)     #
            new_props.append(cand)               #
        # new_props.sort(key=lambda s: s.fitness, reverse=True)
        # print("Best initial proposition fitness:", new_props[0].fitness)
        # self.evaluate_propositions(self.problem.initial_propositions,mode)
        self.problem.initial_propositions.fitness = 0.0
        new_props.append(self.problem.initial_propositions)
        if mode in ("improving", "temp"):
            new_props.sort(
                key=lambda s: (1 if s.fitness > 0 else (0 if s.fitness == 0 else -1),
                            -s.fitness if s.fitness > 0 else s.fitness),
                reverse=True
            )
            print("fitness list:", [new_props[i].fitness for i in range(len(new_props))])
            pool = [p for p in new_props if p.fitness > 0]
            if not pool:
                pool = new_props[:]  #

            weights = [1.0 / (i + 1) for i in range(len(pool))]
            chosen = random.choices(pool, weights=weights, k=1)[0]

        else:
            new_props.sort(key=lambda s: s.fitness, reverse=True)

            pool = new_props
            weights = [1.0 / (i + 1) for i in range(len(pool))]
            chosen = random.choices(pool, weights=weights, k=1)[0]

        print("Chosen proposition fitness2:", chosen.fitness)


        # with open(log_path, "a", encoding="utf-8") as f:
        #     f.write(f"New change_prop-mutation\n")
        # while gen < max_gen:
        #     gen += 1
        #     selected_parents = []
        #     for _ in range(2):
        #         a, b = random.sample(new_props, 2)
        #         selected_parents.append(a if a.fitness > b.fitness else b)
        #     # offspring = crossover_proposition(selected_parents)
        #     offspring = []
        #         for prop in child_props.props:                
        #             prop.mutate(gap)

        #     for ind_props in offspring:
        #         self.evaluate_propositions(ind_props)
        #     new_props.extend(offspring)
        #     new_props.sort(key=lambda s: s.fitness, reverse=True)
        #     new_props = new_props[:num_props]
        for rv in self.problem.reference_vectors:
            rv.other_reward = 0 
        self.problem.variables_propositions = chosen

    def update_ref_reward(self):
        for rv in self.problem.reference_vectors:
            rv.decay_reward(decay_rate=0.25)  #
            rv.association = 0
        for solution in self.solutions:
            if all(obj == 0 for obj in solution.solution_ori_objectives):
                continue
            ref_vector = self.problem.search_ref_vector(solution)
            ref_vector.get_reward(self.problem.reference_vectors)
            ref_vector.association += 1
        for rv in self.problem.reference_vectors:
            self.problem.variables_propositions.fitness += rv.reward_value
        return self.problem.variables_propositions.fitness

    def _maybe_log_progress(self, progress_root: str | None, exp_name: str | None, final: bool = False):
        """MOUT implementation detail."""
        import os, json
        import numpy as np
        from pymoo.indicators.hv import HV

        if not progress_root or not exp_name:
            return

        M = self.problem.number_of_objectives()
        front = np.array([sol.solution_ori_objectives for sol in self.solutions], dtype=float)
        hv = float(HV(ref_point=[0.1] * M)(-front)) if front.size else 0.0

        out_dir = os.path.join(progress_root, "prop_results", str(exp_name))
        os.makedirs(out_dir, exist_ok=True)

        gen = int(getattr(self.termination_criterion, "evaluations", getattr(self, "evaluations", 0)))

        if final:
            for g in range(self.last_gen + 1, 101):
                p = os.path.join(out_dir, f"generations_{g}.txt")
                with open(p, "a", encoding="utf-8") as f:
                    f.write(f"{hv:.4f}, {json.dumps(front.tolist(), ensure_ascii=False)}\n")

            for lo in range(self.last_budget, 300, 10):
                hi = lo + 10
                hv_path = os.path.join(out_dir, f"Budget_{lo:03d}-{hi:03d}.txt")
                with open(hv_path, "a", encoding="utf-8") as f:
                    f.write(f"{hv:.4f}, {json.dumps(front.tolist(), ensure_ascii=False)}, mode={getattr(self, 'current_mode', 'unknown')}\n")
            return

        cur_eval = int(getattr(self, "evaluations", gen))
        cur_bin = cur_eval // 10
        lo, hi = cur_bin * 10, cur_bin * 10 + 10
        self.last_gen = gen
        self.last_budget = hi
        if not hasattr(self, "_logged_bins"):
            self._logged_bins = set()
        if cur_bin not in self._logged_bins:
            hv_path = os.path.join(out_dir, f"Budget_{lo:03d}-{hi:03d}.txt")
            with open(hv_path, "a", encoding="utf-8") as f:
                f.write(f"{hv:.4f}, {json.dumps(front.tolist(), ensure_ascii=False)}, mode={getattr(self, 'current_mode', 'unknown')}\n")
        hv_path2 = os.path.join(out_dir, f"generations_{gen}.txt")
        with open(hv_path2, "a", encoding="utf-8") as f:
            f.write(f"{hv:.4f}, {json.dumps(front.tolist(), ensure_ascii=False)}\n")

        self._logged_bins.add(cur_bin)

class ConfigurationMutation(BitFlipMutation):
    def __init__(self, problem, probability: float):
        super(BitFlipMutation, self).__init__(probability=probability)
        self.problem = problem  #

    def execute(self, solution: NewIntegerSolution) -> NewIntegerSolution:
        for i in range(len(solution.variables)):
            if random.random() <= self.probability:
                candidates = [v for v in self.problem.independent_set[i] if v != solution.variables[i]]
                if candidates:
                    solution.variables[i] = random.choice(candidates)
        return solution

class ConfigurationCrossover(SPXCrossover):
    def __init__(self, probability: float):
        super(ConfigurationCrossover, self).__init__(probability=probability)

    def execute(self, parents: list[NewIntegerSolution]) -> list[NewIntegerSolution]:
        offspring1 = copy.deepcopy(parents[0])
        offspring2 = copy.deepcopy(parents[1])
        print("Crossover executed with probability:", self.probability)
        if random.random() <= self.probability:
            for i in range(len(parents[0].variables)):
                if random.random() < 0.5:
                    offspring1.variables[i] = parents[1].variables[i]
                    offspring2.variables[i] = parents[0].variables[i]
        return [offspring1, offspring2]

__all__ = [
    "MOUT",
    "ConfigDatasetProblem",
    "NewIntegerSolution",
    "ConfigurationMutation",
    "ConfigurationCrossover",
    "ReferenceVector",
]
