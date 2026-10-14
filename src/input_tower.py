"""Targets with controlled input-dependence and Fourier-degree mass towers.

An initial core supplies one function of several inputs. Every subsequent
stage adds terms involving the newly available input and only earlier inputs.
Orthogonality makes each stage's squared mass its exact contribution to the
conditional-expectation input approximation. Degree profiles are independent
of stage mass when max_degree <= core_size.
"""

from itertools import combinations
import math
import random

from src.fourier import FourierFunction, positive_integer
from src.functions import BoolFunction


INPUT_TOWER_FUNCTION_NAMES = (
    "input_tower_increasing", "input_tower_random", "input_tower_top_heavy",
)


class InputTowerFunction(FourierFunction):
    evaluate_input_tower = True

    def __init__(self, input_len=32, max_degree=None, *, core_size=8,
                 degree_profile="increasing", terms_per_stage_degree=16,
                 input_spectrum="powerlaw", input_power=2.0, input_decay=0.9,
                 degree_power=2.0, top_mass=0.9, top_band=None,
                 input_order="prefix", seed=17):
        positive_integer("input_len", input_len)
        positive_integer("core_size", core_size)
        max_degree = core_size if max_degree is None else max_degree
        positive_integer("max_degree", max_degree)
        positive_integer("terms_per_stage_degree", terms_per_stage_degree)
        if core_size > input_len or max_degree > input_len:
            raise ValueError("core_size and max_degree cannot exceed input_len.")
        if degree_profile not in ("increasing", "random", "top_heavy"):
            raise ValueError("Unknown degree profile.")
        if input_spectrum not in ("powerlaw", "geometric"):
            raise ValueError("input_spectrum must be powerlaw or geometric.")
        for name, value in (("input_power", input_power), ("degree_power", degree_power)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not 0 < input_decay < 1:
            raise ValueError("input_decay must be strictly between 0 and 1.")
        if input_order not in ("prefix", "permuted"):
            raise ValueError("input_order must be prefix or permuted.")
        top_band = max(1, math.ceil(max_degree / 4)) if top_band is None else top_band
        positive_integer("top_band", top_band)
        if top_band > max_degree:
            raise ValueError("top_band cannot exceed max_degree.")
        if not 0 < top_mass < 1:
            raise ValueError("top_mass must be strictly between 0 and 1.")
        if top_band < max_degree and top_mass <= top_band / max_degree:
            raise ValueError("top_mass must give the top band more mass per degree than the lower band.")
        BoolFunction.__init__(self, name=f"input_tower_{degree_profile}")
        self.input_len = input_len
        self.max_degree = max_degree
        self.core_size = core_size
        self.degree_profile = degree_profile
        self.terms_per_stage_degree = terms_per_stage_degree
        self.input_spectrum = input_spectrum
        self.input_power = float(input_power)
        self.input_decay = float(input_decay)
        self.degree_power = float(degree_power)
        self.top_mass = float(top_mass)
        self.top_band = top_band
        self.input_order_mode = input_order
        self.seed = seed
        order = list(range(input_len))
        if input_order == "permuted":
            random.Random(seed + 1_000_003).shuffle(order)
        self.input_order = tuple(order)
        stage_ranks = tuple(range(core_size, input_len + 1))
        raw_input_mass = [stage ** (-input_power) if input_spectrum == "powerlaw"
                          else input_decay ** (stage - 1)
                          for stage in range(1, len(stage_ranks) + 1)]
        total = math.fsum(raw_input_mass)
        self.stage_mass = tuple(mass / total for mass in raw_input_mass)
        if any(mass <= 0 for mass in self.stage_mass) or any(
                first <= second for first, second in zip(self.stage_mass, self.stage_mass[1:])):
            raise ValueError("Input-stage mass must be positive and strictly decreasing; use a milder decay.")
        if degree_profile == "increasing":
            raw_degree_mass = [degree ** degree_power for degree in range(1, max_degree + 1)]
        elif degree_profile == "random":
            mass_rng = random.Random(seed + 2_000_003)
            raw_degree_mass = [mass_rng.uniform(0.1, 1.0) for _ in range(max_degree)]
        elif top_band == max_degree:
            raw_degree_mass = [1.0] * max_degree
        else:
            raw_degree_mass = [(1 - top_mass) / (max_degree - top_band)] * (max_degree - top_band)
            raw_degree_mass += [top_mass / top_band] * top_band
        total = math.fsum(raw_degree_mass)
        self.requested_degree_mass = tuple(mass / total for mass in raw_degree_mass)
        if any(not math.isfinite(mass) or mass <= 0 for mass in self.requested_degree_mass):
            raise ValueError("Degree masses must be finite and positive.")

        support_rng = random.Random(seed)
        terms = []
        self.stage_degree_mass = []
        for rank, stage_mass in zip(stage_ranks, self.stage_mass):
            available_degree = min(rank, max_degree)
            # With a smaller core, degree profiles are necessarily conditional
            # on the degrees available at that stage. Record the actual spectrum.
            available_total = math.fsum(self.requested_degree_mass[:available_degree])
            row_mass = [stage_mass * mass / available_total if degree <= available_degree else 0.0
                        for degree, mass in enumerate(self.requested_degree_mass, start=1)]
            self.stage_degree_mass.append(tuple(row_mass))
            for degree in range(1, available_degree + 1):
                if rank == core_size:
                    pool, count_selected, anchor = core_size, degree, ()
                else:
                    pool, count_selected, anchor = rank - 1, degree - 1, (rank - 1,)
                possible = math.comb(pool, count_selected)
                count = min(terms_per_stage_degree, possible)
                if possible <= terms_per_stage_degree:
                    supports = list(combinations(range(pool), count_selected))
                else:
                    selected = set()
                    while len(selected) < count:
                        selected.add(tuple(sorted(support_rng.sample(range(pool), count_selected))))
                    supports = sorted(selected)
                coefficient_size = math.sqrt(row_mass[degree - 1] / count)
                for support in supports:
                    subset = tuple(sorted(order[position] for position in (*support, *anchor)))
                    coefficient = support_rng.choice((-1, 1)) * coefficient_size
                    terms.append((degree, rank, subset, coefficient))
        terms.sort(key=lambda term: (term[0], term[1], term[2]))
        self.subsets = tuple(term[2] for term in terms)
        self.coefficients = tuple(term[3] for term in terms)
        self.degrees = tuple(term[0] for term in terms)
        self.offsets = tuple(sum(degree <= cutoff for degree in self.degrees)
                             for cutoff in range(max_degree + 1))
        self.degree_mass = tuple(math.fsum(row[degree] for row in self.stage_degree_mass)
                                 for degree in range(max_degree))
        self.tail_mass = tuple(math.fsum(self.degree_mass[cutoff:])
                               for cutoff in range(max_degree + 1))
        ranks = {position: rank for rank, position in enumerate(order, start=1)}
        term_ranks = [max(ranks[position] for position in subset) for subset in self.subsets]
        self.input_mass = tuple(math.fsum(coefficient ** 2 for coefficient, rank in
                                         zip(self.coefficients, term_ranks) if rank == cutoff)
                                for cutoff in range(1, input_len + 1))
        self.input_tail_mass = tuple(math.fsum(self.input_mass[cutoff:])
                                    for cutoff in range(input_len + 1))
        self._device_tensors = {}

    def approximation_inputs(self, epsilon):
        """Smallest input-order prefix achieving squared L2 error <= epsilon."""
        if not math.isfinite(epsilon) or epsilon < 0:
            raise ValueError("epsilon must be finite and nonnegative.")
        return next(count for count, tail in enumerate(self.input_tail_mass) if tail <= epsilon)

    def metadata(self):
        independent = self.max_degree <= self.core_size
        per_input_influence = [math.fsum(coefficient ** 2 for subset, coefficient in
                                        zip(self.subsets, self.coefficients) if position in subset)
                               for position in range(self.input_len)]
        return {
            "name": self.metric_name, "family": "input_tower", "layout": "input_tower",
            "spectrum": self.degree_profile, "basis": "chi_S(x)=product(1-2*x_i)",
            "measure": "uniform_independent_binary_bits", "input_len": self.input_len,
            "max_degree": self.max_degree, "core_size": self.core_size,
            "degree_profile": self.degree_profile, "degree_power": self.degree_power,
            "top_mass": self.top_mass, "top_band": self.top_band,
            "degree_profile_scope": "global" if independent else "conditional_on_available_degrees_per_stage",
            "requested_degree_mass": list(self.requested_degree_mass),
            "degree_mass": list(self.degree_mass), "tail_mass_above_degree": list(self.tail_mass),
            "input_order_mode": self.input_order_mode, "input_order": list(self.input_order),
            "input_spectrum": self.input_spectrum, "input_power": self.input_power,
            "input_decay": self.input_decay, "input_stage_sizes": list(range(self.core_size, self.input_len + 1)),
            "input_stage_mass": list(self.stage_mass),
            "input_stage_degree_mass": [list(row) for row in self.stage_degree_mass],
            "input_mass_by_rank": list(self.input_mass), "input_tail_mass_above_count": list(self.input_tail_mass),
            "per_input_influence": per_input_influence,
            "relevant_inputs": sorted(set().union(*map(set, self.subsets))),
            "terms_per_stage_degree": self.terms_per_stage_degree, "seed": self.seed,
            "mean": 0.0, "squared_l2_norm": 1.0,
            "terms_per_degree": [end - start for start, end in zip(self.offsets, self.offsets[1:])],
            "terms": [{"subset": list(subset), "coefficient": coefficient}
                      for subset, coefficient in zip(self.subsets, self.coefficients)],
            "input_approximation": "E[f(X) | X at first k positions of input_order]",
            "input_tower_scope": "Core block, then one additional input per stage; prefix projections are not certified best-k juntas.",
        }


def make_input_tower(name, **kwargs):
    if name not in INPUT_TOWER_FUNCTION_NAMES:
        raise ValueError(f"Unknown input-tower function {name!r}.")
    return InputTowerFunction(degree_profile=name[len("input_tower_"):], **kwargs)
