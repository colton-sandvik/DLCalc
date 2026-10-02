"""Exact equality checks in the localized Soergel bimodule model.

The default verifier reads degree-adaptive pivot entries of F-G directly. It
does not reconstruct coefficients or trust those in an extraction checkpoint.
Vanishing at the nested unisolvent grids, triangularity, and the coefficient
degree bound imply every double-leaf coefficient vanishes.

For small problems ``method='matrix'`` checks every column symbolically through
the elementary generator graphs. This avoids the pivot/leaf theorem entirely
and is a useful independent regression oracle. Neither method is randomized.
Over the supported characteristic-zero root realizations, localization is
faithful on Bott--Samelson bimodules. This verifies a bimodule identity; it does
not assume the missing H3 presentation relation in order to prove that relation.
"""

from __future__ import annotations

from time import monotonic

from formats import check_integer, require
from leaves import DoubleLeavesBasis
from morphisms import Localization, Morphism
from quadratic import encode_scalar
from reconstruct import Expansion, interpolation_grid


def is_zero_symbolic(
    morphism,
    basis,
    *,
    leaf_method="weighted",
    cache_size=20000,
    progress=None,
    progress_every=10000,
):
    """
    Check if a homogeneous morphism is zero by calculating the lambda functionals symbolically.

    Parameters
    ----------
    morphism: Morphism
        A homogeneous morphism of degree basis.degree, or the zero morphism. 
    basis: DoubleLeavesBasis
        A DoubleLeavesBasis with the same system and boundaries as morphism.
    leaf_method: optional
        The method for computing the lambda functionals on double leaves. Can be either "weighted" for the weight
        dot product formula or "Elias-Williamson graphs" for the default formula for computing the lambda functional
        of a general Elias-Williamson graph.
    cache_size: int, optional
        Size of cache used for storing repeated calculations. Number is in stored calls, not bytes.
    progress: callable, optional
        Optional callback receiving a dictionary of progress information.
        For example, progress=print.
    progress_every: int, optional
        Number of double leaf indices between progress reports.

    Returns
    -------
    a boolean indicating whether `morphism` = 0

    Notes
    -----
    All localization arithmetic takes place in Frac(R).

    The function computes the lambda functionals symbolically.
    """

    started = monotonic()
    degree = basis.degree
    system = basis.system

    if morphism.system is not system:
        raise ValueError(
            "The morphism and basis must share one system."
        )

    if (morphism.source, morphism.target) != (basis.source, basis.target):
        raise ValueError("The morphism and basis have different boundaries.")

    if any(d != degree for d in morphism.homogeneous_components):
        raise ValueError(
            f"The morphism must be homogeneous of degree {degree}."
        )

    progress_every = check_integer(progress_every, "progress interval")
    if progress_every <= 0:
        raise ValueError("progress_every must be positive.")
   
    evaluator = Localization(system, cache_size=cache_size, leaf_method=leaf_method)
    for i in range(len(basis)):
        value = basis.lambda_value(morphism, i, evaluator)
        if value:
            leaf = basis[i]
            if progress is not None:
                progress(f"The morphism is nonzero. The ({leaf.target_mask}, {leaf.source_mask})-entry of the localization matrix (lambda_{i}) is {value}.")
            return False
        
        # Print out a progress report if needed
        if progress is not None and (
            (i + 1) % progress_every == 0
            or i + 1 == len(basis)
        ):
            progress({
                "degree" : degree,
                "candidate": i + 1,
                "candidates": len(basis),
                "seconds": monotonic() - started,
            })

    return True
    

def is_zero_interpolate(
    morphism,
    basis,
    *,
    leaf_method="weighted",
    cache_size=20000,
    progress=None,
    progress_every=10000,
):
    """
    Check if a homogeneous morphism is zero by calculating the lambda functionals via interpolation.

    Parameters
    ----------
    morphism: Morphism
        A homogeneous morphism of degree basis.degree, or the zero morphism. 
    basis: DoubleLeavesBasis
        A DoubleLeavesBasis with the same system and boundaries as morphism.
    leaf_method: optional
        The method for computing the lambda functionals on double leaves. Can be either "weighted" for the weight
        dot product formula or "Elias-Williamson graphs" for the default formula for computing the lambda functional
        of a general Elias-Williamson graph.
    cache_size: int, optional
        Size of cache used for storing repeated calculations. Number is in stored calls, not bytes.
    progress: callable, optional
        Optional callback receiving a dictionary of progress information.
        For example, progress=print.
    progress_every: int, optional
        Number of double leaf indices between progress reports.

    Returns
    -------
    a boolean indicating whether `morphism = 0`

    Notes
    -----
    All localization arithmetic takes place in Frac(R).

    The function computes the lambda functionals using polynomial interpolation.
    """

    started = monotonic()
    degree = basis.degree
    system = basis.system
    kk = system.field

    if morphism.system is not system:
        raise ValueError(
            "The morphism and basis must share one system."
        )

    if (morphism.source, morphism.target) != (basis.source, basis.target):
        raise ValueError("The morphism and basis have different boundaries.")

    if any(d != degree for d in morphism.homogeneous_components):
        raise ValueError(
            f"The morphism must be homogeneous of degree {degree}."
        )

    progress_every = check_integer(progress_every, "progress interval")
    if progress_every <= 0:
        raise ValueError("progress_every must be positive.")
   
    
    if not len(basis):
        return True
        
    grid = interpolation_grid(system.rank, max(basis.coefficient_degrees)) 
    for number, (level, raw_point) in enumerate(grid):

        # The evaluation point
        point = tuple(kk(x) for x in raw_point)

        # Make all localization arithmetic occur at the specified point
        evaluator = Localization(system, point=point, cache_size=cache_size, leaf_method=leaf_method)
        
        # We only need to check lambda_i for those i with deg(DL_i) >= 2 * level 
        indices = [i for i, k in enumerate(basis.coefficient_degrees) if k >= level]

        count = 0
        for i in indices:
            value = basis.lambda_value(morphism, i, evaluator)
            if value:
                leaf = basis[i]
                if progress is not None:
                    progress(f"The morphism is nonzero. The ({leaf.target_mask}, {leaf.source_mask})-entry of the localization matrix (lambda_{i}) at {point} is {encode_scalar(value, kk)}.")
                return False
        
            # Print out a progress report if needed
            if progress is not None and (
                (count + 1) % progress_every == 0
                or count + 1 == len(indices) 
            ):
                progress({
                    "degree" : degree,
                    "level": level,
                    "point": number,
                    "total points": len(grid), 
                    "candidate": count + 1,
                    "candidates": len(indices),
                    "seconds": monotonic() - started,
                })

            count += 1

    return True

    

def is_zero_matrix(
    morphism,
    *,
    leaf_method="weighted",
    cache_size=20000,
    progress=None,
    progress_every=10000,
):
    """
    Check if a morphism is zero by computing its localization matrix.

    Parameters
    ----------
    morphism: Morphism
    leaf_method: optional
        The method for computing the localization matrix of double leaves. Can be either "weighted" for the weight
        dot product formula or "Elias-Williamson graphs" for the default formula for computing the localization matrix
        of a general Elias-Williamson graph.
    cache_size: int, optional
        Size of cache used for storing repeated calculations. Number is in stored calls, not bytes.
    progress: callable, optional
        Optional callback receiving a dictionary of progress information.
        For example, progress=print.
    progress_every: int, optional
        Number of column indices between progress reports.

    Returns
    -------
    a boolean indicating whether `morphism` = 0

    Notes
    -----
    All localization arithmetic takes place in Frac(R).

    This method is substantially slower than using either of the other methods. Should only be used for 
    morphisms with small localization matrices.
    """

    started = monotonic()
    system = morphism.system
    evaluator = Localization(system, cache_size=cache_size, leaf_method=leaf_method)
    number_of_columns = 1 << len(morphism.source)

    # Loop through every column of the localization matrix
    for i in range(number_of_columns):
        column = evaluator.column(morphism, i)
        # Check if any entries of the column is nonzero
        for j, entry in column.items():
            if entry:
                if progress is not None:
                    progress(f"Morphism is nonzero. The ({j}, {i})-entry of the localization matrix is {entry}.")
                return False

        # Print out a progress report if needed
        if progress is not None and (
            (i + 1) % progress_every == 0
            or i + 1 == number_of_columns
        ):
            progress({
                "current column": i + 1,
                "number of columns": number_of_columns,
                "seconds": monotonic() - started,
            })
    
    return True

def equal(
    left,
    right,
    *,
    algorithm="interpolate",
    leaf_method="weighted",
    cache_size=20000,
    progress=None,
    progress_every=10000,
):
    """
    Compare two homogeneous morphism `left` and `right` of the same degree 
    by computing the lambda functionals of both sides using interpolation.

    Parameters
    ----------
    left: Morphism 
    right: Morphism 
    algorithm: optional
        The algorithm to be used to verify equality. Can be one of the following options:
        - "interpolate" : determine equality be computing the lambda functionals using interpolation 
        - "symbolic" : determine equality by computing the lambda functions symbolically
        - "matrix" : determine equality by computing the localization matrices
    leaf_method: optional
        The method for computing the entries for the localization matrix of double leaves. 
        Can be either "weighted" for the weight dot product formula or "Elias-Williamson graphs" 
        for the default formula for computing the localization entries of a general Elias-Williamson graph.
    cache_size: int, optional
        Size of cache used for storing repeated calculations. Number is in stored calls, not bytes.
    progress: callable, optional
        Optional callback receiving a dictionary of progress information.
        For example, progress=print.
    progress_every: int, optional
        Number of double leaf indices between progress reports.
        If the algorithm is "matrix", then this it is the number of column indices between progress reports.

    Returns
    -------
    a boolean indicating whether `left = right`

    Notes
    -----
    The "matrix" algorithm should only be used for morphisms with small boundaries. The algorithm is much
    more computationally expensive than the other methods. 
    """
    
    require(leaf_method in ("weighted", "Elias-Williamson graphs"), f"{leaf_method} is an invalid leaf method.")
    require(algorithm in ("matrix", "symbolic", "interpolate"), f"{algorithm} is an invalid algorithm for equality verification.")

    started = monotonic()

    if left.system is not right.system:
        if progress is not None:
            progress("Underlying Coxeter group and realization of both morphisms are different.")
        return False
    system = left.system

    if left.source != right.source:
        if progress is not None:
            progress(f"Source expression for left {left.source} is not equal to the source expression for right {right.source}.")
        return False
    source = left.source

    if left.target != right.target:
        if progress is not None:
            progress(f"Target expression for left {left.target} is not equal to the target expression for right {right.target}.")
        return False
    target = left.target

    difference = left-right
    
    # If the algorithm is "matrix", then we can directly call iz_zero_matrix
    if algorithm == "matrix":
        if progress is not None:
            progress(f"Starting verification using the localization matrix algorithm.")
        flag = is_zero_matrix(difference, leaf_method=leaf_method, cache_size=cache_size, progress=progress, progress_every=progress_every)
        if progress is not None:
            progress(f"Verification completed in {monotonic() - started} seconds. Morphisms are {"equal" if flag else "not equal"}.")
        return flag

    # If the algorithm is "symbolic" or "interpolate", then we need to first break up `difference` into homogeneous components
    homogeneous_components = sorted(difference.homogeneous_components.items())

    for degree, component in homogeneous_components:
        basis = DoubleLeavesBasis(system, source, target, degree)
        if progress is not None:
            progress(f"Double leaf basis computed in degree {degree}. There are {len(basis)} contributing double leaves.")

        # If the degree has no nonzero matrices, skip the check
        if not len(basis):
            continue

        comp_flag = True
        if algorithm == "symbolic":
            if progress is not None:
                progress(f"Starting verification of degree {degree} component using the symbolic algorithm.")
            comp_flag = is_zero_symbolic(difference, basis, leaf_method=leaf_method, cache_size=cache_size, progress=progress, progress_every=progress_every)
            
        elif algorithm == "interpolate":
            if progress is not None:
                progress(f"Starting verification of degree {degree} component using the interpolation algorithm.")
            comp_flag = is_zero_interpolate(difference, basis, leaf_method=leaf_method, cache_size=cache_size, progress=progress, progress_every=progress_every)
            
        # if comp_flag is False, then component is nonzero
        if not comp_flag:
            if progress is not None:
                progress(f"Verification completed in {monotonic() - started}. Degree {degree} components are not equal.")
            return False
    
    if progress is not None:
        progress(f"Verification completed in {monotonic() - started} seconds. Morphisms are equal.")
    return True
