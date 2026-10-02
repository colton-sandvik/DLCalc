"""Some helpers for doing arithmetic in coefficient fields and encoding coefficients when saving results

All arithmetic happens in sage: in particular, ``golden_field()`` is the number
field Q(phi), with phi^2 = phi + 1 and the positive real embedding. 

The other supported computational coefficient fields are QQ and Sage's exact
real algebraic field AA. AA permits the geometric realization of any finite
rank Coxeter matrix, including edges with arbitrary finite labels. 
One should avoid using AA when possible since it can be substantially slower than QQ or Q(phi).
"""

from __future__ import annotations

from functools import lru_cache
from sage.all import AA, QQ, NumberField, PolynomialRing

from formats import require


@lru_cache(maxsize=1)
def golden_field():
    """Return Sage's exact Q(phi), using the positive root of t^2-t-1."""
    t = PolynomialRing(QQ, "t").gen()
    return NumberField(t*t - t - 1, "phi", embedding=(AA(1) + AA(5).sqrt())/2)


def phi():
    """Return the distinguished generator phi of Q(phi) = :func:`golden_field`."""
    return golden_field().gen()


def field_data(field) -> dict:
    """Describe one of the three supported coefficient fields."""
    if field is QQ:
        return {"kind": "QQ"}
    if field is AA:
        return {"kind": "AA"}
    if field is golden_field():
        return {"kind": "Qphi"} 
    raise ValueError("Use QQ, Qphi, or AA as the coefficient field.")


def field_from_data(data: dict):
    """Return a field from a coefficient-field descriptor of the form {kind: field_kind}
    
    Example:
    `field_from_data ({'kind': 'Qphi}) = golden_field()`
    """
    require(isinstance(data, dict), "Invalid coefficient field.")
    for field in (QQ, golden_field(), AA):
        if data == field_data(field):
            return field
    raise ValueError("Unknown coefficient-field descriptor.")


def _rational(text):
    """Converts a rational number encoded as a string into an element of QQ
    """
    require(isinstance(text, str), "Rationals must be encoded as strings.")
    parts = text.split("/")
    require(len(parts) <= 2 and all(p.lstrip("-").isdigit() for p in parts), "Invalid rational string.")
    return QQ(text)


def encode_scalar(value, field) -> dict:
    """Encode a scalar in QQ, QQ(phi), AA.

    The exact methodology of encoding the scalar depends on the coefficient field.
    For QQ, the result is a dict {`q`: str(v)}
    For QQ(phi), the result is a dict {'a': str(a), 'b': str(b)} where value = a + b * phi
    For AA, the result is a dict {'minimal_polynomial': min_poly, 'real_root_index': roots_index} 
        here min_poly is the coefficient list for the minimal polynomial of value
        here roots_index is the index of value in the list of the real roots (ordered in increasing order)

    Examples:
    `encode_scalar(5/7, Q(phi)) = {'q': '5/7'}
    
    `encode_scalar(4+3phi, Q(phi)) = {'a': '4', 'b': 3}

    `encode_scalar(sqrt(2)+sqrt(3), AA) = {`minimal_polynomial': ['1', '0', '-10', '0', '1'], 'real_root_index': 3}
        In this case, the minimal polynomial is x^4 - 10x^2 + 1
        The roots are [-sqrt(3) - sqrt(2), -sqrt(3) + sqrt(2), sqrt(3) - sqrt(2), sqrt(3) + sqrt(2)]
        So the minimal polynomial has coefficients [1,0,-10,0,1] and sqrt(2) + sqrt(3) has index 3 in the list of roots
    """
    value = field(value)
    if field is QQ:
        return {"q": str(value)}
    if field is golden_field():
        coordinates = list(value)
        coordinates += [QQ(0)]*(2-len(coordinates))
        return {"a": str(coordinates[0]), "b": str(coordinates[1])}
    elif field is AA:
        polynomial = value.minpoly()
        roots = sorted(polynomial.roots(AA, multiplicities=False))
        return {"minimal_polynomial": [str(c) for c in polynomial.list()],
            "real_root_index": roots.index(value)}
    else:
        raise ValueError("Only scalars in QQ, QQ(phi), or AA are supported")



def decode_scalar(data: dict, field):
    """Decode a scalar using the format produced from :func:`encode_scalar`
    
    This function is inverse to that of :func:`encode_scalar`.
    """
    require(isinstance(data, dict), "Invalid scalar object.")
    if set(data) == {"q"}:
        return field(_rational(data["q"]))
    if set(data) == {"a", "b"}:
        require(field is golden_field(), "A Q(phi) scalar requires its declared field.")
        return _rational(data["a"]) + _rational(data["b"])*phi()
    elif field is AA and set(data) == {"minimal_polynomial", "real_root_index"}:
        polynomial = PolynomialRing(QQ, "t")([_rational(c) for c in data["minimal_polynomial"]])
        require(polynomial.degree() >= 1 and polynomial.is_irreducible(), "Expected an irreducible minimal polynomial.")
        roots = sorted(polynomial.roots(AA, multiplicities=False))
        index = int(data["real_root_index"])
        require(0 <= index < len(roots), "Real root index is out of range.")
        return roots[index]
    else:
        raise ValueError("Only scalars in QQ, QQ(phi), or AA are supported")


def encode_polynomial(polynomial) -> list:
    """Encode a univariate or multivariate polynomial in terms of its exponents and coefficients.
    If multivariate the exponents are ordered in the declared variable order. Zero returns the empty list.

    Examples:
    `encode_polynomial(x^4 - 10x^2 + 1) = [
    {`powers`: [0]: 'coefficient': {'q': 1}},
    {`powers`: [2]: 'coefficient': {'q': -10}},
    {`powers`: [4]: 'coefficient': {'q': 1}},
    ]
    """
    field = polynomial.parent().base_ring()
    encoded = []
    kv_pairs = sorted(polynomial.dict().items())
    for powers, value in kv_pairs:
        if value:
            if isinstance(powers, int):
                encoded.append({"powers": [powers], "coefficient": encode_scalar(value, field)})
            else:
                encoded.append({"powers": [int(e) for e in powers], "coefficient": encode_scalar(value, field)})
    return encoded


def decode_polynomial(data: list, ring):
    """Read a polynomial, rejecting duplicate monomials and negative powers.
    
    This is the inverse to :func:`encode_polynomial`.
    """
    require(isinstance(data, list), "Expected list of monomials.")
    values = {}
    for record in data:
        require(isinstance(record, dict) and set(record) == {"powers", "coefficient"}, "Invalid monomial record.")
        powers = tuple(int(e) for e in record["powers"])
        require(len(powers) == ring.ngens() and all(e >= 0 for e in powers), "Invalid monomial powers.")
        key = powers[0] if len(powers) == 1 else powers
        require(key not in values, "Duplicate polynomial monomial.")
        values[key] = decode_scalar(record["coefficient"], ring.base_ring())

    return ring(values)


def homogeneous_parts(polynomial) -> dict:
    """Return nonzero homogeneous pieces keyed by *ordinary* (i.e., not the degree in R) polynomial degree. 
    This is a dict whose keys are degrees and whose values are elements of the parent ring of `polynomial`
    
    Examples:
    `homogenous_parts(xy + y^2 + 4x + 3y) = {1: 4x + 3y, 2: xy + y^2}`
    `homogenous_parts(x^4 - 10x^2 +1) = {0: 1, 2: -10, 4: 1}
    """
    ring = polynomial.parent()
    pieces = {}
    for powers, value in polynomial.dict().items():
        if isinstance(powers, int):
            degree = powers
        else:
            degree = sum(powers)
        pieces.setdefault(degree, {})[powers] = value
    return {int(degree): ring(coefficients) for degree, coefficients in sorted(pieces.items())}

# This seems pointless. Going to exclude for now
# def rational_coordinates(value, field) -> tuple:
#     """Return coordinates in Z or Z[phi] for an integrality audit.

#     Integrality over arbitrary algebraic integer rings is deliberately not
#     inferred from an AA encoding. Such audits need an explicit integral model.
#     """
#     if field is QQ:
#         return (QQ(value),)
#     if field is golden_field():
#         data = encode_scalar(value, field)
#         return (QQ(data["a"]), QQ(data["b"]))
#     raise NotImplementedError("Integral descent is implemented for explicit QQ and Q(phi) models only.")
