"""
Find the coefficients of the double leaf expansion for a morphism.

There are two implemented algorithms based on the companion paper.
The first makes use of symbolic arithmetic which is cleaner, but slow.
The second makes use of interpolation to make all arithmetic take place 
in the ground field.

A coefficient expansion is still not confirmed until independently checked by verify.py.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from time import monotonic

from sage.all import matrix, vector, QQ

from coxeter import (
    CoxeterSystem, 
    monomials, 
    polynomial_dimension
)

from formats import (
    sha256, 
    check_integer, 
    read_json, 
    require, 
    write_json
)

from leaves import (
    DominanceIndex, 
    DoubleLeavesBasis, 
    bit_indices
)

from morphisms import (
    Localization, 
    Morphism
)

from quadratic import (
    decode_polynomial, 
    decode_scalar, 
    encode_polynomial, 
    encode_scalar, 
    homogeneous_parts,
    golden_field
)


@dataclass(frozen=True)
class ExpansionTerm:
    """An explicit polynomial coefficient times one chosen double leaf."""

    source_mask: int
    target_mask: int
    coefficient: object

    def __post_init__(self):
        object.__setattr__(self, "source_mask", check_integer(self.source_mask, "source mask"))
        object.__setattr__(self, "target_mask", check_integer(self.target_mask, "target mask"))

    @property
    def key(self):
        return self.source_mask, self.target_mask


class Expansion:
    """A finite sum in the fixed double-leaves basis, with computation metadata.

    It is not an equality certificate. An incomplete expansion is still a
    well-defined finite sum; :meth:`as_morphism` denotes exactly that sum.
    Repeated leaf keys are combined; zero coefficients are omitted.
    """

    def __init__(self, system, source, target, terms=(), *, metadata=None):
        self.system = system
        self.source, self.target = system.word(source), system.word(target)
        self.metadata = dict(metadata or {})

        combined = {}
        for term in terms:
            e = system.subexpression(self.source, term.source_mask)
            f = system.subexpression(self.target, term.target_mask)
            require(e.endpoint == f.endpoint, "Expansion masks have different endpoints.")
            combined[term.key] = combined.get(term.key, system.ring.zero())+system.ring(term.coefficient)
        self.terms = tuple(ExpansionTerm(e, f, c) for (e, f), c in combined.items() if c)

    @cached_property
    def support_index(self):
        """Reusable dominance support, independent of an evaluation point."""
        return DominanceIndex(self.system, self.source, self.target, (t.key for t in self.terms))

    def as_morphism(self):
        """The finite sum itself, with sparse entry evaluation and shared leaves."""
        return Morphism(self.system, self.source, self.target, "expansion", self)

    def homogeneous_components(self):
        """Homogeneous morphism components, including polynomial coefficient degrees."""
        pieces = defaultdict(list)
        for term in self.terms:
            d = (self.system.subexpression(self.source, term.source_mask).defect
                 + self.system.subexpression(self.target, term.target_mask).defect)
            for k, coefficient in homogeneous_parts(term.coefficient).items():
                pieces[d+2*k].append(ExpansionTerm(*term.key, coefficient))
        return {d: Expansion(self.system, self.source, self.target, terms).as_morphism()
                for d, terms in pieces.items()}

    def dual(self):
        """Reflect every double leaf, retaining left polynomial coefficients."""
        return Expansion(self.system, self.target, self.source,
                         [ExpansionTerm(t.target_mask, t.source_mask, t.coefficient) for t in self.terms])

    def to_body(self):
        terms = []
        for term in self.terms:
            e = self.system.subexpression(self.source, term.source_mask)
            f = self.system.subexpression(self.target, term.target_mask)
            terms.append({"source_mask": term.source_mask, "target_mask": term.target_mask,
                          "middle_word": list(self.system.reduced_word(e.endpoint)),
                          "leaf_degree": e.defect+f.defect, "coefficient": encode_polynomial(term.coefficient)})
        return {"terms": terms, "metadata": self.metadata}

    def to_data(self):
        """A coefficient file containing all realization and basis data."""
        return {"system": self.system.to_data(), "source": list(self.source), "target": list(self.target),
                **self.to_body()}

    def is_integral(self):
        """
        Check if the coefficients are integral
        """

        for term in self.terms:
            coeff = term.coefficient
            if not all(a.is_integral() for a in coeff.coefficients()):
                return False
        
        return True

    @classmethod
    def from_body(cls, data, system, source, target):
        require(isinstance(data.get("terms"), list), "Invalid coefficient body.")
        source, target = system.word(source), system.word(target)
        terms, seen = [], set()
        for record in data["terms"]:
            e, f = check_integer(record["source_mask"], "source mask"), check_integer(record["target_mask"], "target mask")
            require((e, f) not in seen, "Duplicate double-leaf key.")
            seen.add((e, f))
            a, b = system.subexpression(source, e), system.subexpression(target, f)
            require(a.endpoint == b.endpoint, "Double-leaf endpoints differ.")
            require(system.word(record["middle_word"]) == system.reduced_word(a.endpoint), "Incorrect canonical middle word.")
            require(int(record["leaf_degree"]) == a.defect+b.defect, "Incorrect double-leaf degree.")
            coefficient = decode_polynomial(record["coefficient"], system.ring)
            require(bool(coefficient), "Zero terms must be omitted.")
            terms.append(ExpansionTerm(e, f, coefficient))
        return cls(system, source, target, terms, metadata=data.get("metadata"))

    @classmethod
    def from_data(cls, data, *, system=None):
        system = system or CoxeterSystem.from_data(data["system"])
        require(system.to_data() == data["system"], "The supplied realization differs from the file.")
        return cls.from_body(data, system, data["source"], data["target"])

    def save(self, path):
        """Atomically write the coefficient format."""
        return write_json(path, self.to_data())

    @classmethod
    def load(cls, path, *, system=None):
        """Load expansion from a JSON file."""
        return cls.from_data(read_json(path), system=system)


def interpolation_grid(rank, maximum_degree):
    """Return (level, point) on nested translated simplex grids.

    For rank 3 the first points are (2,3,1), (2,4,1), (3,3,1).
    Positive coordinates are regular for the positive geometric realization.
    Rank 1 uses the single point (1), sufficient for every homogeneous degree.
    """
    if rank == 1:
        return ((0, (1,)),)
    return tuple((level, tuple(2+i+e for i, e in enumerate(powers))+(1,))
                 for level in range(maximum_degree+1) for powers in monomials(level, rank-1))


def interpolate(system, degree, points, values):
    """Exactly interpolate one homogeneous polynomial; no numerical fitting."""
    exponents = monomials(degree, system.rank)
    require(len(points) == len(values) == len(exponents), "Interpolation sample count is incorrect.")
    rows = [[_monomial_value(point, powers) for powers in exponents] for point in points]
    coefficients = matrix(system.field, rows).solve_right(vector(system.field, values))
    return system.ring({powers: c for powers, c in zip(exponents, coefficients) if c})


def _monomial_value(point, powers):
    value = 1
    for coordinate, exponent in zip(point, powers):
        value *= coordinate**exponent
    return value


def solve_coefficients_symbolic(
    morphism,
    basis,
    *,
    cache_size=20000,
    progress=None,
    progress_every=1000,
):
    """
    Compute polynomial coefficients in a fixed double-leaves basis using the symbolic algorithm.

    Parameters
    ----------
    morphism: Morphism
        A homogeneous morphism of degree basis.degree, or the zero morphism. 
        The double leaf expansion will be found for this morphism.
    basis: DoubleLeavesBasis
        A DoubleLeavesBasis with the same system and boundaries as morphism.
    cache_size: int, optional
        Size of cache used for storing repeated calculations. Number is in stored calls, not bytes.
    progress: callable, optional
        Optional callback receiving a dictionary of progress information.
        For example, progress=print.
    progress_every: int, optional
        Number of coefficients between progress reports.

    Returns
    -------
    tuple
        Polynomial coefficients in basis order, including zero coefficients.

    Notes
    -----
    All localization arithmetic takes place in Frac(R). Each solved
    coefficient is converted back to R immediately.

    The function computes only the entries needed for triangular
    elimination; it does not construct the full localization matrix.

    The algorithm can be found in the companion paper. 
    """
    system = basis.system
    R = system.ring
    evaluator = Localization(system, cache_size=cache_size)

    if morphism.system is not system:
        raise ValueError(
            "The morphism and basis must share one system."
        )

    if (morphism.source, morphism.target) != (basis.source, basis.target):
        raise ValueError("The morphism and basis have different boundaries.")

    if any(d != basis.degree for d in morphism.homogeneous_components):
        raise ValueError(
            f"The morphism must be homogeneous of degree {basis.degree}."
        )

    progress_every = check_integer(progress_every, "progress interval")
    if progress_every <= 0:
        raise ValueError("progress_every must be positive.")

    index = DominanceIndex(
        system,
        basis.source,
        basis.target,
        (pair.key for pair in basis.entries)
    )

    coefficients = []

    nonzero_count = 0
    started = monotonic()

    for i, pair in enumerate(basis.entries):
        # residual is the running sum for coefficient[i]
        # This is computed by the formula
        # coefficient[i] = (lambda_i (morphism) - sum_(j < i) coefficient[j] * lambda_i (DL_j)) / lambda_i (DL_i)
        residual = basis.lambda_value(morphism, i, evaluator) # lambda_i (morphism)

        # Only the pairs (e,f) below (pair.source_mask, pair.target_mask) have nonzero lambda_i's.
        contributors = index.contributors(pair.source_mask, pair.target_mask)

        for j in contributors:
            if j < i and coefficients[j] != 0:
                # lambda_i (DL_j)
                entry = evaluator.entry(basis.morphism(j), pair.target_mask, pair.source_mask,)
                if entry != 0:
                    residual -= coefficients[j] * entry

        # Once the entire off-diagonal sum has been subtracted,
        # a zero residual gives a zero coefficient without division.
        if not residual:
            coefficient = R.zero()
        else:
            # lambda_i (DL_i)
            diagonal = basis.diagonal(i, evaluator)

            # Check whether lambda_i (DL_i) = 0
            # This is not mathematically possible, so something has gone very wrong if this error is called.
            if not diagonal:
                raise ArithmeticError(
                    f"The symbolic diagonal vanished at basis index {i}."
                )

            # The i-th coefficient before being coerced from Frac(R) to R
            quotient = residual / diagonal

            # Try to coerce quotient to an element of R. 
            # if this fails, then something has gone very wrong, since it is mathematically impossible
            # for quotient to not be an element of R
            try:
                coefficient = R(quotient)
            except (TypeError, ValueError) as error:
                raise ArithmeticError(
                    f"Coefficient {i} = {quotient}, with masks {pair.key}, "
                    "does not belong to the polynomial ring."
                ) from error

            # Roots have categorical degree 2. These are ordinary
            # polynomial degrees, already computed by DoubleLeavesBasis.
            # We check that coefficient[i] is homogeneous and has the expected degree.
            # If this fails, then it likely means that this function was called on a non-homogeneous morphism
            expected_degree = basis.coefficient_degrees[i]
            if any(
                sum(powers) != expected_degree
                for powers in coefficient.dict()
            ):
                raise ArithmeticError(
                    f"Coefficient {i} is not homogeneous of ordinary "
                    f"polynomial degree {expected_degree}."
                )

        # add the coefficient to our list of coefficients
        coefficients.append(coefficient)

        # increase the running count of nonzero coefficients
        if coefficient:
            nonzero_count += 1

        # create a process report 
        if progress is not None and (
            (i + 1) % progress_every == 0
            or i + 1 == len(basis)
        ):
            progress({
                "degree": basis.degree,
                "candidate": i + 1,
                "candidates": len(basis),
                "nonzero_coefficients": nonzero_count,
                "seconds": monotonic() - started,
            })

    return tuple(coefficients)


def solve_coefficients_interpolate(
    morphism,
    basis,
    *,
    cache_size=20000,
    progress=None,
    progress_every=1000,
):
    """
    Compute polynomial coefficients in a fixed double-leaves basis using the interpolation algorithm.

    Parameters
    ----------
    morphism: Morphism
        A homogeneous morphism of degree basis.degree, or the zero morphism. 
        The double leaf expansion will be found for this morphism.
    basis: DoubleLeavesBasis
        A DoubleLeavesBasis with the same system and boundaries as morphism.
    cache_size: int, optional
        Size of cache used for storing repeated calculations. Number is in stored calls, not bytes.
    progress: callable, optional
        Optional callback receiving a dictionary of progress information.
        For example, progress=print.
    progress_every: int, optional
        Number of coefficients between progress reports.

    Returns
    -------
    tuple
        Polynomial coefficients in basis order, including zero coefficients.

    Notes
    -----
    All localization arithmetic takes place in the ground field k. Interpolation
    is then used to convert coefficients evaluated at a finite collection of points
    to elements of R.

    The function computes only the entries needed for triangular
    elimination; it does not construct the full localization matrix.

    The algorithm can be found in the companion paper. 
    """

    system = basis.system
    R = system.ring
    kk = system.field

    if morphism.system is not system:
        raise ValueError(
            "The morphism and basis must share one system."
        )

    if (morphism.source, morphism.target) != (basis.source, basis.target):
        raise ValueError("The morphism and basis have different boundaries.")


    if any(d != basis.degree for d in morphism.homogeneous_components):
        raise ValueError(
            f"The morphism must be homogeneous of degree {basis.degree}."
        )

    progress_every = check_integer(progress_every, "progress interval")
    if progress_every <= 0:
        raise ValueError("progress_every must be positive.")
    
    if not len(basis):
        return ()

    index = DominanceIndex(
        system,
        basis.source,
        basis.target,
        (pair.key for pair in basis.entries)
    )
    
    known = {}
    pairs = basis.entries
    degrees = basis.coefficient_degrees
    grid = interpolation_grid(system.rank, max(degrees))
    histories = [[] for _ in pairs]
    sampled_points = []

    started = monotonic()

    for number, (level, raw_point) in enumerate(grid):

        # The evaluation point
        point = tuple(kk(x) for x in raw_point)

        # Make all localization arithmetic occur at the specified point
        evaluator = Localization(system, point=point, cache_size=cache_size)

        # List of coefficients evaluated at the specified point
        values = []
        active = 0

        for i, pair in enumerate(pairs):
            # Check if c_i is determined by interpolation already
            if i in known:
                # c_i (point)
                value = kk(known[i](*point))
            else:
                
                # lambda_i (DL_i) (point)
                residual = basis.lambda_value(morphism, i, evaluator)

                # Only the pairs (e,f) below (pair.source_mask, pair.target_mask) have nonzero lambda_i's.
                contributors = index.contributors(pair.source_mask, pair.target_mask)

                for j in contributors:
                    if j < i and values[j] != 0:
                        # lambda_i (DL_j) (point)
                        entry = evaluator.entry(basis.morphism(j), pair.target_mask, pair.source_mask,)
                        if entry != 0:
                            residual -= values[j] * entry
                
                # Once the entire off-diagonal sum has been subtracted,
                # a zero residual gives a zero coefficient without division.
                if not residual:
                    value = kk(0)
                else:
                    # lambda_i (DL_i) (point)
                    diagonal = basis.diagonal(i, evaluator)

                    # Check whether lambda_i (DL_i) (point) = 0
                    # This can only happen if the point is not regular.
                    if not diagonal:
                        raise ArithmeticError(
                            f"The diagonal vanished at basis index {i} with point {point}."
                        )

                    # The i-th coefficient before being coerced from Frac(R) to R
                    value = residual / diagonal
                
            values.append(value)

            # create a process report 
            if progress is not None and (
                (i + 1) % progress_every == 0
                or i + 1 == len(pairs)
            ):
                progress({
                    "level": level,
                    "total levels": max(degrees),
                    "point": number,
                    "total points": len(grid),
                    "candidate": i + 1,
                    "candidates": len(basis),
                    "seconds": monotonic() - started,
                })
        
        # Check whether the calculated point is enough to interpolate some of the polynomials.
        # Add these interpolated polynomials to the `known` dictionary.
        sampled_points.append(point)
        for i, value in enumerate(values):
            histories[i].append(value)
            needed = polynomial_dimension(degrees[i], system.rank)
            if i not in known and len(histories[i]) == needed:
                known[i] = interpolate(system, degrees[i], sampled_points[:needed], histories[i])

    # Flatten the coefficients into a tuple with the same index as `basis`.
    return tuple(known[i] for i in sorted(known))
    

def reconstruct(morphism, *, cache_size=20000, method="interpolate", progress=None, progress_every=1000):
    """Find the polynomial double-leaf coefficients of a general morphism.

    Parameters
    ----------
    morphism : Morphism
        The morphism that the double leaves expansion will be found for. Does not need to be homogeneous.
    cache_size: int, optional
        Size of cache used for storing repeated calculations. Number is in stored calls, not bytes.
    method : string, optional
        Either "interpolate" or "symbolic". Specifies whether the symbolic or interpolation argument should be used.
        Defaults to the interpolation algorithm.
    progress: callable, optional
        Optional callback receiving a dictionary of progress information.
        For example, progress=print.
    progress_every: int, optional
        Number of coefficients between progress reports.

    Returns an :class:`Expansion`. Run ``verify.equal(F, result.as_morphism())``
    for an independent equality check. 
    """

    system = morphism.system
    terms = []
    started = monotonic()
    profiles = []

    for degree, component in sorted(
        morphism.homogeneous_components.items()
    ):
        local_started = monotonic()
        if progress:
            progress(f"Computation started for the degree {degree} homogeneous component.")
        basis = DoubleLeavesBasis(
            system,
            morphism.source,
            morphism.target,
            degree=degree,
        )
        if progress:
            progress(f"Double leaf basis found for degree {degree} morphisms. Total number of candidate double leaves is {len(basis.entries)}")
        if method == "symbolic":
            if progress:
                progress(f"Symbolic coefficient solve started for degree {degree} homogeneous component.")
            coefficients = solve_coefficients_symbolic(
                component,
                basis,
                cache_size=cache_size,
                progress=progress,
                progress_every=progress_every,
            )
        elif method == "interpolate":
            if progress:
                progress(f"Interpolation coefficient solve started for degree {degree} homogeneous component.")
            coefficients = solve_coefficients_interpolate(
                component,
                basis,
                cache_size=cache_size,
                progress=progress,
                progress_every=progress_every,
            )
        else:
            raise ValueError("Only options for method are either 'symbolic' or 'interpolate'. ")

        if progress:
            progress(f"Coefficient solve completed for degree {degree} homogeneous component. Took {monotonic() - local_started} seconds to complete.")

        terms.extend(
            ExpansionTerm(
                pair.source_mask,
                pair.target_mask,
                coefficient,
            )
            for pair, coefficient in zip(basis.entries, coefficients)
            if coefficient
        )

        profile = {
            "degree": degree,
            "candidates": len(basis.entries),
            "seconds": monotonic() - local_started
        }
        profiles.append(profile)

    # Expansion combines contributions to the same double leaf
    # arising from different homogeneous components.
    return Expansion(
        system,
        morphism.source,
        morphism.target,
        terms,
        metadata={
            "algorithm": method,
            "seconds": monotonic() - started,
            "input_check_sum": sha256(morphism.to_data()), 
            "profile" : profiles
        },
    )