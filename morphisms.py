"""Construct formal Bott--Samelson morphisms and evaluate their localization.

Morphism nodes store generators, sums, left polynomial multiples, compositions,
and tensor products without evaluating matrices. ``F.then(G)`` and ``G @ F``
both mean G after F. Polynomial boxes may occupy any horizontal region.

Localization computes sparse columns on demand in the normalization of Elias
and Williamson (EW23), *Localized calculus for the Hecke category*, Section 2
(https://doi.org/10.5802/ambp.415). The braid formula is Lemma 2.11 in the
balanced setting. CoxeterSystem checks balance and finite-edge quantum
conditions; this module relies on those checks.
Rows index target subexpressions, columns index source subexpressions, and bit
zero records the leftmost strand. Coefficients are exact rational functions or
exact values at a point. Sparse evaluation avoids constructing a full matrix,
although a single column, and especially a braid block, can still be large.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from functools import cached_property, lru_cache
from types import MappingProxyType
from numbers import Integral

from coxeter import CoxeterSystem, mask_to_subexpression
from formats import check_integer, read_json, require, write_json
from quadratic import decode_polynomial, encode_polynomial, homogeneous_parts

GENERATOR_DEGREES = {"upper dot": 1, "lower dot": 1, "merge": -1, "split": -1,
                     "cap": 0, "cup": 0, "braid": 0}


@dataclass(frozen=True)
class Operation:
    """Immutable record for a generator padded by identity strands on either side.

    ``position`` is the first consumed strand, or an insertion gap/box region
    between 0 and the current word length. Lower dots and cups supply ``color``;
    braids supply ``length``; boxes supply ``polynomial``. Other colors come from
    the current source word. 
    """

    kind: str
    position: int
    color: object = None
    length: int | None = None
    polynomial: object = None

    def __post_init__(self):
        """Normalize integral positions, lengths, and labels to Python ints; reject booleans."""
        object.__setattr__(self, "position", check_integer(self.position, "generator position"))
        if self.length is not None:
            object.__setattr__(self, "length", check_integer(self.length, "braid length"))
        if isinstance(self.color, Integral):
            object.__setattr__(self, "color", check_integer(self.color, "generator label"))

    def to_data(self):
        """Return a dictionary version of the data.

        Omit absent optional fields and encode a polynomial as exact monomial data.
        The polynomial must already belong to a supported polynomial ring.
        """
        data = {"kind": self.kind, "position": self.position}
        if self.color is not None:
            data["color"] = self.color
        if self.length is not None:
            data["length"] = self.length
        if self.polynomial is not None:
            data["polynomial"] = encode_polynomial(self.polynomial)
        return data

    @classmethod
    def from_data(cls, data, system):
        """Decode a generator record using ``system.ring`` for its polynomial.

        This is the inverse to :func:`to_data`
        """
        require(isinstance(data, dict) and {"kind", "position"} <= set(data)
                <= {"kind", "position", "color", "length", "polynomial"},
                "Invalid generator record.")
        position = check_integer(data["position"], "generator position")
        length = check_integer(data["length"], "braid length") if "length" in data else None
        polynomial = decode_polynomial(data["polynomial"], system.ring) if "polynomial" in data else None
        return cls(data["kind"], position, data.get("color"), length, polynomial)


def operation_target(system, word, operation):
    """Find the target of a generated surrounded by identity strands.
     ``word`` is assumed to have been constructed by ``system.word``, this is the source of the morphism. 

    Return ``(target_word, consumed, inserted)``.
        - target_word is the target of the morphism
        - consumed is the number of terms removed from the source word
        - inserted is the number of terms inserted from the source word    
    """
    kind, i = operation.kind, operation.position
    require(isinstance(i, int) and not isinstance(i, bool) and 0 <= i <= len(word), f"Invalid generator position. Position {i} must be between {0} and {len(word)}")
    if kind == "polynomial":
        require(operation.polynomial is not None and operation.color is None and operation.length is None,
                "Invalid polynomial box.")
        system.ring(operation.polynomial)
        return word, 0, 0
    require(kind in GENERATOR_DEGREES and operation.polynomial is None, f"Unknown generator kind. Generator provided is of type {kind}.")
    if kind == "lower dot":
        require(operation.color in system.labels and operation.length is None, "Inserted color is missing or invalid.")
        target = word[:i] + (operation.color, ) + word[i:] # the source s_1 ... s_i ... s_k has target s_1 ... s_i s ... s_k where s=color
        return target, 0, 1 # target has inserted color at position i, no simples of the source were consumed, but one simple was inserted
    elif kind == "cup":
        require(operation.color in system.labels and operation.length is None, "Inserted color is missing or invalid.")
        target = word[:i] + (operation.color, operation.color) + word[i:] 
        return target, 0, 2 # target has inserted (color, color) at position i, no simples of the source were consumed, but two simples were inserted
    elif kind == "merge":
        require(operation.color is None and operation.length is None, "Merge generator gets its color from its source word.")
        require(i+2 <= len(word), "Generator extends beyond its boundary.")
        require(word[i] == word[i+1], f"Merge generator is a one-color generator. The word {word} has different colors at positions {i} and {i+1}.")
        target = word[:i] + (word[i], ) + word[i+2:]
        return target, 2, 1 # target replaces (color, color) at position (i, i+1) by (color,). Two simples of the source were consumed, but one was inserted
    elif kind == "cap":
        require(operation.color is None and operation.length is None, "Cap generator gets its color from its source word.")
        require(i+2 <= len(word), f"The word {word} has length {len(word)}. Cap generator cannot exist at positions ({i}, {i+1}).")
        require(word[i] == word[i+1], f"Cap generator is a one-color generator. The word {word} has different colors at positions {i} and {i+1}.")
        target = word[:i] + word[i+2:]
        return target, 2, 0 # target removes (color, color) at position (i, i+1). Two simples of the source were consumed, but none were inserted
    elif kind == "split":
        require(operation.color is None and operation.length is None, "Split generator gets its color from its source word.")
        require(i+1 <= len(word), f"The word {word} has length {len(word)}. Split generator cannot exist at position {i}")
        target = word[:i] + (word[i], word[i]) + word[i+1:]
        return target, 1, 2 # target replaces (color, ) at position i by (color, color). One simple of the source was consumed, and two were inserted
    elif kind == "upper dot":
        require(operation.color is None and operation.length is None, "Upper dot generator gets its color from its source word.")
        require(i+1 <= len(word), f"The word {word} has length {len(word)}. Upper dot generator cannot exist at position {i}")
        target = word[:i] + word[i+1:]
        return target, 1, 0 # target removes (color, ) at position i. One simple of the source was consumed, and none were inserted
    elif kind == "braid":
        m = operation.length
        require(operation.color is None, "Braid generator gets its colors from its source word.")
        require(isinstance(m, int) and m >= 2, f"Braid generators requires m={m} >= 2")
        require(i+m <= len (word), f"The word {word} has length {len(word)}, and the braid is of length {m}. Braid generator cannot exist at position {i}")
        s, t = word[i:i+2]
        require(s != t and system.braid_order(s, t) == m, f"Braid length disagrees with the Coxeter matrix. For simples {s} and {t} we have m = {system.braid_order(s,t)} when it should be {m}")
        require(word[i:i+m] == tuple(s if j % 2 == 0 else t for j in range(m)), f"Braid colors {s} and {t} do not alternate in {word} from positions {i} to {i+m-1}.")
        replacement = tuple(t if j % 2 == 0 else s for j in range(m))
        target = word[:i]+replacement+word[i+m:]
        return target, m, m # target replaces (s,t,s,...) at positions i through i+m-1  by (t,s,t,...). m simples consumed and m simple inserted


@dataclass(frozen=True, eq=False)
class Morphism:
    """
    A formal morphism B_source -> B_target in Soergel bimodules

    A morphism is stored as a directed tree. Each node of the tree can be of the following types:
        - zero - a zero morphism
        - Elias-Williamson graph - a single Elias-Williamson or Soergel graph. The graph is stored as a sequence of Operations. I.e., it is a composition of generators of the form:
            - zero morphisms
            - identity morphisms
            - polynomial boxes : B_emptyset -> B_emptset
            - upper dots : B_s -> B_emptyset
            - lower dots : B_emptyset -> B_s
            - merge: B_ss -> B_s
            - splits : B_s -> B_ss
            - 2m-valent vertices: B_sts... -> B_tst...
        - sum - this node always has at least two children, say f and g, and the node represents f + g
        - scale - this node always has a scalar, c, associated to it and one child, say f. The node represents cf 
        - compose - this node always has two children, say f and g, and the node represents g o f
        - tensor - this node always has two children, say f and g, and the node represents f⊗g
        - double_leaf - a single double leaf morphism. The only data stored on this node other than the source and target is the source and target subexpressions
        - expansion - a linear combination of double leaves along with some metadata. This is used for representing the double-leaf expansion of a fixed morphism

    Construct the class through the named methods. Use :func:`verify.equal` for an efficient equality check.

    Nodes share children rather
    than distributing tensor products or compositions over sums. ``*`` denotes
    multiplication by a polynomial in the left exterior region; for a right
    polynomial action use ``box(..., region=len(word))`` explicitly.

     This class is mostly a convenient black box for storing the information of a morphism in a convenient manner.
    Localization data for these morphisms are not automatically computed.
    """

    system: CoxeterSystem
    source: tuple
    target: tuple
    kind: str
    payload: object = None
    children: tuple = ()

    @classmethod
    def identity(cls, system, word=()):
        """Identity morphism of the Bott--Samelson object for ``word`` (possibly nonreduced)."""
        word = system.word(word)
        return cls(system, word, word, "Elias-Williamson graph", ())

    @classmethod
    def zero(cls, system, source, target):
        """Zero morphism with specified boundaries."""
        return cls(system, system.word(source), system.word(target), "zero")

    @classmethod
    def elias_williamson_graph(cls, system, source, operations):
        """Build an Elias--Williamson graph corresponding to a sequence of operations read from bottom to top.

        Freeze ``operations`` into a tuple, fix the source, and walk through the
        operations with ``operation_target`` to validate and derive the final target.
        Every operation position refers to its own current boundary.
        """
        source = system.word(source)
        operations = tuple(operations)
        word = source
        normalized = []
        for operation in operations:
            word, _, _ = operation_target(system, word, operation)
            if operation.kind == "polynomial":
                operation = replace(operation, polynomial=system.ring(operation.polynomial))
            normalized.append(operation)
        return cls(system, source, word, "Elias-Williamson graph", tuple(normalized))

    @classmethod
    def generator(cls, system, source, kind, position=0, *, color=None, length=None):
        """Insert one dot, merge, split, cup, cap, or braid in ``source``.

        Validate the position and, for a braid without ``length``, infer the finite
        Coxeter order of the two adjacent colors. Store a one-operation graph, which
        checks all remaining boundary constraints. Use ``box`` for polynomial boxes.
        """
        source = system.word(source)
        position = check_integer(position, "generator position")
        require(0 <= position <= len(source), "Invalid generator position.")
        if kind == "braid" and length is None:
            require(position+1 < len(source), "A braid needs two adjacent colors.")
            length = system.braid_order(*source[position:position+2])
            require(length is not None, "There is no finite braid vertex on an infinite edge.")
        return cls.elias_williamson_graph(system, source, [Operation(kind, position, color, length)])

    @classmethod
    def box(cls, system, word, polynomial, region=0):
        """Place a polynomial in region ``0..len(word)``, counted from the left.

        Coerce it into ``system.ring`` and store a boundary-preserving operation.
        Localization acts on the polynomial by the endpoint of the selected strands
        to its left, so different regions generally give different maps.
        """
        return cls.elias_williamson_graph(system, word, [Operation("polynomial", region, polynomial=system.ring(polynomial))])

    @classmethod
    def upper_dot(cls, system, color):
        """The degree-one upper dot morphism B_color -> B_empty."""
        source = (color,)
        return cls.elias_williamson_graph(system, source, [Operation("upper dot", 0)])

    @classmethod
    def lower_dot(cls, system, color):
        """The degree-one lower dot morphism B_empty -> B_color."""
        source = ()
        return cls.elias_williamson_graph(system, source, [Operation("lower dot", 0, color)])

    @classmethod
    def merge(cls, system, color):
        """The degree-minus-one merge morphism B_(color,color) -> B_color."""
        source = (color, color)
        return cls.elias_williamson_graph(system, source, [Operation("merge", 0)])

    @classmethod
    def cup(cls, system, color):
        """The degree-zero cup morphism B_empty -> B_(color,color)."""
        source = ()
        return cls.elias_williamson_graph(system, source, [Operation("cup", 0, color)])

    @classmethod
    def split(cls, system, color):
        """The degree-minus-one split morphism B_color -> B_(color,color)."""
        source = (color,)
        return cls.elias_williamson_graph(system, source, [Operation("split", 0)])

    @classmethod
    def cap(cls, system, color):
        """The degree-zero cap morphism B_(color,color) -> B_empty."""
        source = (color, color)
        return cls.elias_williamson_graph(system, source, [Operation("cap", 0)])

    @classmethod
    def braid(cls, system, color1, color2):
        """The degree-zero 2m-valent vertex (braid morphism) on two distinct colors.

        Look up their finite Coxeter order m and build the alternating source tuple
        of length m beginning with color1. The target begins with color2. 
        """
        length = system.braid_order(color1, color2)
        require(length is not None, "There is no finite braid vertex on an infinite edge.")
        require(color1 != color2, "A braid needs two distinct colors.")
        source = tuple(color1 if i % 2 == 0 else color2 for i in range(length))
        return cls.elias_williamson_graph(system, source, [Operation("braid", 0, length=length)])
    

    def append(self, kind, position=0, *, color=None, length=None):
        """Compose with one generator on the current target boundary at ``position``.

        Construct that generator with ``generator`` and then compose to self.
        """
        return self.then(Morphism.generator(self.system, self.target, kind, position, color=color, length=length))

    def _compatible(self, other):
        """Quick check to make sure two morphisms have the same underlying Coxeter system and realization."""
        require(isinstance(other, Morphism) and other.system is self.system,
                "Morphisms must share one CoxeterSystem instance; pass system= when loading files.")

    def then(self, other):
        """Vertical composition: apply self first, then other.

        Checks that intermediate boundaries match and that the Coxeter system and realization are shared. 
        An explicit zero factor gives zero; two Elias-Williamson graphs are concatenated and validated.
        Otherwise retain the two factors as children of a compose node.
        """
        self._compatible(other)
        require(self.target == other.source, "Vertical boundaries do not match.")
        if self.kind == "zero" or other.kind == "zero":
            return Morphism.zero(self.system, self.source, other.target)
        if self.kind == other.kind == "Elias-Williamson graph":
            return Morphism.elias_williamson_graph(self.system, self.source, self.payload+other.payload)
        return Morphism(self.system, self.source, other.target, "compose", children=(self, other))

    def __matmul__(self, other):
        """Binary operation for morphism composition.
         ``self @ other`` means ``other.then(self)`` (self after other).
         """
        return other.then(self)

    def tensor(self, other):
        """Horizontal concatenation of morphisms. 
        Place self to the left of other and concatenate their boundary words.

        Checks that both morphisms have the same Coxeter system and realization.
        Result is a morphism of kind `tensor` with two children. 
        No distribution across sums occurs, and no localization matrices are computed automatically.
        """
        self._compatible(other)
        return Morphism(self.system, self.source+other.source, self.target+other.target,
                        "tensor", children=(self, other))

    def __add__(self, other):
        """Binary operation for sum of morphisms.
        Checks that maps have the same boundaries and the same Coxeter system and realizaiton.

        Accept integral zero for Python's ``sum``, discard explicit zero nodes, and
        flatten immediate sum children into one sum node. Does not combine equivalent
        terms or detect cancellation of a map with its negative.
        """
        if isinstance(other, Integral) and other == 0:
            return self
        self._compatible(other)
        require(self.source == other.source and self.target == other.target, "Sum boundaries do not match.")
        if self.kind == "zero":
            return other
        if other.kind == "zero":
            return self
        left = self.children if self.kind == "sum" else (self,)
        right = other.children if other.kind == "sum" else (other,)
        return Morphism(self.system, self.source, self.target, "sum", children=left+right)

    def __radd__(self, other):
        """Right addition always uses ``__add__`` instead. 
        
        This is only really relevant when working with something like 0 + f"""
        return self.__add__(other)

    def scale(self, polynomial):
        """Multiply by a polynomial in the left exterior region.

        Coerce polynomial to the ambient ring, simplify scalar zero/one and explicit zero maps,
        and combine nested scalar multiples. Otherwise store one child and the
        polynomial in a scale node. Both multiplication operators use this method.
        """
        polynomial = self.system.ring(polynomial)
        if not polynomial or self.kind == "zero":
            return Morphism.zero(self.system, self.source, self.target)
        if polynomial == 1:
            return self
        if self.kind == "scale":
            return self.children[0].scale(polynomial*self.payload)
        return Morphism(self.system, self.source, self.target, "scale", polynomial, (self,))

    __mul__ = scale # This means that f*c should be interpreted as cf. Multiplication is always on the left
    __rmul__ = scale 

    def __neg__(self):
        """Return the negation of a morphism."""
        return self.scale(-1)

    def __sub__(self, other):
        """Subtract morphisms by adding the negation."""
        return self + (-other)

    def dual(self):
        """
        Takes the dual of the morphism, i.e., the horizontal reflection in the Bott--Samelson category
        Source and target of morphism are exchanged.

        For Elias-Williamson graphs, we traverse the operations in the opposite order. 
        For each operation (i.e., generator) the dual is given by
            - dual (upper dot) = lower dot
            - dual (lower dot) = upper dot
            - dual (merge) = split
            - dual (split) = merge
            - dual (cup) = cap
            - dual (cap) = cup
            - dual (braid) = braid (with the roles of s and t swapped)
            - dual (identity) = identity
            - dual (polynomial) = polynomial 
        
        For sums: dual (f + g) = dual (f) + dual (g)
        For scale: dual (cf) = c dual (f)
        For compositions: dual (f o g) = dual (g) o dual (f)
        For tensors: dual (f ⊗ g) = dual (f) ⊗ dual (g)
        For double leaves: dual (LL_(e,f)) = LL_(f,e)
        For expansion: we simply take the dual of each term in the expansion
        """
        if self.kind == "zero":
            return Morphism.zero(self.system, self.target, self.source)
        if self.kind == "Elias-Williamson graph":
            word, operations = self.source, []
            inverse_kind = {"upper dot": "lower dot", "lower dot": "upper dot", "merge": "split", "split": "merge", "cap": "cup", "cup": "cap"}
            for op in self.payload:
                if op.kind in ("braid", "polynomial"):
                    inverse = op
                else:
                    kind = inverse_kind[op.kind]
                    color = word[op.position] if kind in ("lower dot", "cup") else None
                    inverse = Operation(kind, op.position, color=color)
                operations.append(inverse)
                word, _, _ = operation_target(self.system, word, op)
            return Morphism.elias_williamson_graph(self.system, self.target, reversed(operations))
        if self.kind == "sum":
            return sum(child.dual() for child in self.children)
        if self.kind == "scale":
            return self.children[0].dual().scale(self.payload)
        if self.kind == "compose":
            first, second = self.children
            return second.dual().then(first.dual())
        if self.kind == "tensor":
            return self.children[0].dual().tensor(self.children[1].dual())
        if self.kind == "double_leaf":
            from leaves import double_leaf # only loads double leaf package if double leaves appear, otherwise `morphism.py` can be used without it
            source_mask, target_mask = self.payload
            return double_leaf(self.system, self.target, self.source, target_mask, source_mask)
        if self.kind == "expansion":
            return self.payload.dual().as_morphism()
        raise ValueError(f"Unknown morphism node: {self.kind}")

    @cached_property
    def homogeneous_components(self):
        """Return a cached read-only mapping from degrees to homogenous morphisms of that degree.

        Traverse generator sequences (Elias-Williamson graphs) and split polynomial boxes into ordinary homogeneous
        polynomials, doubling their degrees to match the degree of the realization ring. 
        Group sums by degree; add degrees together for composition/tensor and add twice the scalar polynomial degree for scaling.
        Double-leaf degree is the sum of the two defects; expansions already have their degrees as metadata
        payload. Zero morphism returns an empty mapping.

        This is a formal decomposition: relations and additive cancellations are
        not resolved, so a piece can represent a mathematically zero map.
        """
        def add(result, degree, value):
            """Accumulate one homogeneous map in its degree bucket using formal addition."""
            result[degree] = result[degree]+value if degree in result else value

        if self.kind == "zero":
            return MappingProxyType({})
        if self.kind == "Elias-Williamson graph":
            pieces = {0: Morphism.identity(self.system, self.source)}
            for op in self.payload:
                next_pieces = {}
                options = [(GENERATOR_DEGREES[op.kind], op)] if op.kind != "polynomial" else [
                    (2*d, Operation("polynomial", op.position, polynomial=p))
                    for d, p in homogeneous_parts(op.polynomial).items()]
                for degree, value in pieces.items():
                    for extra, operation in options:
                        add(next_pieces, degree+extra, value.then(Morphism.elias_williamson_graph(self.system, value.target, [operation])))
                pieces = next_pieces
            return MappingProxyType(pieces)
        result = {}
        if self.kind == "sum": # for sums, you treat each component individually
            for child in self.children:
                for d, value in child.homogeneous_components.items():
                    add(result, d, value)
        elif self.kind == "scale": # A polynomial of ordinary degree k adds 2*k.
            for d, value in self.children[0].homogeneous_components.items():
                for k, polynomial in homogeneous_parts(self.payload).items():
                    add(result, d+2*k, value.scale(polynomial))
        elif self.kind in ("compose", "tensor"): # the degree of a tensor is the sum of the degrees
            for d, first in self.children[0].homogeneous_components.items():
                for e, second in self.children[1].homogeneous_components.items():
                    value = first.then(second) if self.kind == "compose" else first.tensor(second)
                    add(result, d+e, value)
        elif self.kind == "double_leaf": # the degree of a double leaf is given by the sum of the defects for the subexpressions defining it
            ea, eb = self.payload
            degree = self.system.subexpression(self.source, ea).defect+self.system.subexpression(self.target, eb).defect
            result[degree] = self
        elif self.kind == "expansion": # expansion already have their homogenous components built in as metadata
            result.update(self.payload.homogeneous_components())
        else:
            raise ValueError(f"Unknown morphism node: {self.kind}")
        return MappingProxyType(result)

    @property
    def degree(self):
        """Return the degree if homogeneous (before cancellations), or None if the morphism is not homogeneous.

        Computes ``homogeneous_components``. Explicit zero and multiple degrees return
        None, but a formally homogeneous expression that cancels can still have a
        degree here; this property does not test mathematical nonvanishing.
        """
        degrees = tuple(self.homogeneous_components)
        if len (degrees) == 1: # check if homogenous
            return degrees[0]
        else:
            return None

    def lambda_value(self, basis, index, *, point=None, evaluator=None):
        """Computes the indexed lambda functional  with respect to a double leaves basis ``basis``.

        If no evaluator is supplied, creates a Localization instance at ``point`` which
        uses interpolation arithmetic as a backend. If no point is supplied, it uses
        symbolic arithmetic. 
        
        Wrapper for ``basis.lambda_value(self, index, evaluator)``. A supplied
        evaluator takes precedence over a given point. 

        The ``index`` refers to which lambda functional of ``self`` to compute
        with respect to the supplied basis. The calculation of the lambda functional
        can be found in the ``DoubleLeavesBasis`` class.
        """
        if evaluator is None:
            evaluator = Localization(self.system, point=point)
        return basis.lambda_value(self, index, evaluator)

    def to_data(self):
        """Serialize the morphism and Coxeter system to JSON-compatible data.

        Depth-first traversal visits children before parents and records each node
        once by object identity. Child indices therefore point backward in the node
        array, preserving shared references. Encode generator/polynomial payloads
        explicitly; expansions provide their own JSON representation.
        """
        nodes, seen = [], {}

        def visit(value):
            """Emit a node after its children, reusing its index if already visited."""
            if value in seen:
                return seen[value]
            children = [visit(child) for child in value.children]
            node = {"kind": value.kind, "source": list(value.source), "target": list(value.target)}
            if children:
                node["children"] = children
            if value.kind == "Elias-Williamson graph":
                node["operations"] = [op.to_data() for op in value.payload]
            elif value.kind == "scale":
                node["polynomial"] = encode_polynomial(value.payload)
            elif value.kind == "double_leaf":
                node["source_mask"], node["target_mask"] = value.payload
            elif value.kind == "expansion":
                node["expansion"] = value.payload.to_body()
            seen[value] = len(nodes)
            nodes.append(node)
            return seen[value]

        root = visit(self)
        return {"system": self.system.to_data(), "nodes": nodes, "root": root}

    @classmethod
    def from_data(cls, data, *, system=None):
        """Reconstruct a JSONified morphism and check that it is valid.

        This is the inverse of ``to_data``.
        """
        system = system or CoxeterSystem.from_data(data["system"])
        require(system.to_data() == data["system"], "The supplied system differs from the file.")
        nodes = []
        require(isinstance(data["nodes"], list), "Invalid node array.")
        for record in data["nodes"]:
            source, target = system.word(record["source"]), system.word(record["target"])
            references = [check_integer(i, "node reference") for i in record.get("children", [])]
            require(all(0 <= i < len(nodes) for i in references), "Nodes must refer only to preceding nodes.")
            children = [nodes[i] for i in references]
            kind = record["kind"]
            if kind == "Elias-Williamson graph":
                require(not children, "An Elias-Williamson graph has no children.")
                value = cls.elias_williamson_graph(system, source, [Operation.from_data(op, system) for op in record["operations"]])
            elif kind == "zero":
                require(not children, "Zero has no children.")
                value = cls.zero(system, source, target)
            elif kind == "sum":
                require(bool(children), "A sum needs children.")
                value = sum(children)
            elif kind == "scale":
                require(len(children) == 1, "A scalar multiple needs one child.")
                value = children[0].scale(decode_polynomial(record["polynomial"], system.ring))
            elif kind in ("compose", "tensor"):
                require(len(children) == 2, "A binary operation needs two children.")
                value = children[0].then(children[1]) if kind == "compose" else children[0].tensor(children[1])
            elif kind == "double_leaf":
                from leaves import double_leaf # Only loads double leaves if needed. 
                require(not children, "A double-leaf reference has no children.")
                value = double_leaf(system, source, target, check_integer(record["source_mask"]), check_integer(record["target_mask"]))
            elif kind == "expansion":
                from reconstruct import Expansion # Only uses expansion from `reconstruct.py` if needed.
                require(not children, "An expansion reference has no children.")
                value = Expansion.from_body(record["expansion"], system, source, target).as_morphism()
            else:
                raise ValueError(f"Unknown morphism node: {kind}")
            require(value.source == source and value.target == target, "Recorded node boundaries disagree with its construction.")
            nodes.append(value)
        root = check_integer(data["root"], "root node")
        require(0 <= root < len(nodes), "Root node is out of range.")
        return nodes[root]

    def save(self, path):
        """Encode with ``to_data`` and atomically write JSON; return the output Path."""
        return write_json(path, self.to_data())

    @classmethod
    def load(cls, path, *, system=None):
        """Read JSONified copy of a morphism and load the morphism using ``from_data``."""
        return cls.from_data(read_json(path), system=system)


    def equal(self, other):
        """
        Check if two morphisms are equal by computing their localization matrices.
        This is a very expensive operation and should not be used on large morphisms.
        """
        
        if not isinstance(other,Morphism):
            return NotImplemented

        if self.system != other.system:
            return False 
        
        evaluator = Localization(other.system)
        loc_mat_1 = evaluator.localization_matrix(self)
        loc_mat_2 = evaluator.localization_matrix(other)
        return loc_mat_1 == loc_mat_2



class NonRegularPointError(ValueError):
    """A root requested during localization evaluates to zero.

    Localization formulas often need a root evaluated at a given point to be invertible. 
    Regularity is checked as needed,and ``root`` rejects zero even when that particular call is not a denominator;
    this is not a test of every root of the Coxeter system.
    """


class Localization:
    """Evaluate sparse normalized localization symbolically or at an exact point.

    With ``point=None``, coefficients lie in the fraction field of system.ring, and all arithmetic is symbolic.
    Otherwise coerce one coordinate per simple root to system.field and evaluate
    there. This is used for the interpolation version of the double-leaf expansion algorithm.
    
    Each instance owns seven LRU caches, each limited to ``cache_size`` entries
    (not bytes). This helps to cache commonly used localization computations.
    
    The default ``leaf_method='weighted'`` uses the optional formula for localization entries of double-leaves
    in terms of a weight dot product of localization entries of light-leaves. There is also an 
    optional implementation of ``leaf_method='Elias-Williamson graphs' which computes the localization entries
    by decomposing a double leaf into its constituent generators.
    """

    def __init__(self, system, point=None, *, cache_size=100000, leaf_method="weighted"):
        """Select the exact coefficient domain, validate options, and install caches.

        Check the leaf-method name, positive integral cache size, and point dimension.
        Wrap bound helper methods with per-instance LRU caches. No morphism is evaluated
        and point regularity is not checked until a root is requested.
        """
        require(leaf_method in ("weighted", "Elias-Williamson graphs"), "Unknown light-leaf evaluation method.")
        cache_size = check_integer(cache_size, "cache size")
        require(cache_size > 0, "Cache size must be positive.")
        self.system = system
        self.leaf_method = leaf_method
        self.point = None if point is None else tuple(system.field(x) for x in point)
        require(self.point is None or len(self.point) == system.rank, "Point has the wrong dimension.")
        self.domain = system.ring.fraction_field() if self.point is None else system.field
        self.leaf_method = leaf_method
        self.root = lru_cache(maxsize=cache_size)(self.root)
        self.zeta = lru_cache(maxsize=cache_size)(self.zeta)
        self.polynomial = lru_cache(maxsize=cache_size)(self.polynomial)
        self._braid = lru_cache(maxsize=cache_size)(self._braid)
        self._column = lru_cache(maxsize=cache_size)(self._column)
        self._double_leaf = lru_cache(maxsize=cache_size)(self._double_leaf)
        self._expansion_index = lru_cache(maxsize=cache_size)(self._expansion_index)

    def __repr__(self):
        if not self.point:
            return f"Symbolic localization evaluator for {self.system}. Double leaves are computed using the {self.leaf_method} algorithm."
        else:
            return f"Interpolation localization evaluator for {self.system} using the point {self.point}. Double leaves are computed using the {self.leaf_method} algorithm."

    def polynomial(self, polynomial, w):
        """Return w(polynomial) where w is an element of the Coxeter group and polynomial is an element of self.system.ring;
        Optionally evaluates the result at the configured point ``self.point``.

        Method uses an LRU cache to speed up repeated computations.
        """
        value = self.system.act(w, polynomial)
        return self.domain(value if self.point is None else value(*self.point))

    def root(self, w, color):
        """ The exact value of w * alpha_color. 
        If ``self.point`` is specified, then returns the evaluation at that point; otherwise, returns the full polynomial.

        Raises an error if the result is zero.

        Method uses an LRU cache to speed up repeated computations.
        """
        j = self.system._positions[color]
        M = self.system.action_matrix(w)
        variables = self.system.ring.gens() if self.point is None else self.point
        result = self.domain(sum(M[i, j]*variables[i] for i in range(self.system.rank)))
        if not result:
            raise NonRegularPointError(f"The root {self.system.reduced_word(w)}({color}) vanishes at {self.point}.")
        return result

    def zeta(self, word, mask, w):
        """The input is a word (s_1, ..., s_k) and a mask which corresponds to a subexprression (e_1, ..., e_k). 
        From this, one can define the Bruhat stroll w_1, ..., w_k where w_i = s_1^e_1 ... s_i^e_i
        Define beta_i = w_(i-1) (alpha_(s_i)). Then the result of the function is the product of the beta_i's. 

        Method uses an LRU cache to speed up repeated computations.
        """
        product = self.domain.one()
        ue = mask_to_subexpression(mask, len(word))
        for i in range(len(word)):
            product *= self.root(w, word[i]) # computes w_(i-1) alpha_(s_i)
            if ue[i]: # check if e_i = 1
                w = self.system.multiply_simple(w, word[i]) # defines w_i
        return product

    def _braid(self, word, prefix):
        """Construct the localization matrix of a local 2m-valent vertex.

        This implements Lemma 2.11 of EW23 for balanced realizations. 
        Returned tuples contain (row_mask, coefficient) pairs per column.
        Enumeration is exponential in the braid length.

        Method uses LRU cache to speed up repeated computations.
        """
        target = tuple(word[1] if i % 2 == 0 else word[0] for i in range(len(word)))
        pi = self.zeta(word, (1 << len(word))-1, prefix)
        by_endpoint = defaultdict(list)
        for mask in range(1 << len(word)):
            endpoint = self.system.subexpression(target, mask).endpoint
            by_endpoint[endpoint].append((mask, pi/self.zeta(target, mask, prefix)))
        return tuple(tuple(by_endpoint[self.system.subexpression(word, mask).endpoint])
                     for mask in range(1 << len(word)))

    def apply_operation(self, word, operation, vector, prefix):
        """Apply one positioned generator to a sparse vector and return (target, output).

        ``vector`` maps valid source masks to coefficients in the evaluator domain;
        ``word`` is a normalized tuple and ``prefix`` is a group element. Validate the
        operation, then split each mask into untouched left bits, consumed local bits,
        and untouched right bits. Multiply prefix by the endpoint of the left bits
        and use that action on the local generator coefficients (or regional box).
        Replace the local bits with target bits, shift the suffix to its new position,
        and accumulate equal output masks, removing zero totals. Output is a mutable
        dictionary.         
        """
        target, consumed, inserted = operation_target(self.system, word, operation)
        i, kind = operation.position, operation.kind
        output = defaultdict(self.domain.zero)
        for mask, value in vector.items():
            low = mask & ((1 << i)-1)
            local = (mask >> i) & ((1 << consumed)-1)
            high = mask >> (i+consumed)
            actual_prefix = self.system.multiply(prefix, self.system.subexpression(word[:i], low).endpoint)
            if kind == "polynomial":
                output[mask] += self.polynomial(operation.polynomial, actual_prefix)*value
                continue

            if kind == "braid":
                # Braid coefficients use Lemma 2.11 of EW23.
                transitions = self._braid(word[i:i+consumed], actual_prefix)[local]

            elif kind == "upper dot":
                color = word[i]
                alpha = self.root(actual_prefix, color)
                # Localization matrix of upper dot is
                #     |   0   | 1
                # -------------
                # * | alpha | 0
                if local == 0:
                    transitions = ((0, alpha),)
                elif local == 1:
                    transitions = ()

            elif kind == "lower dot":
                # Localization matrix of lower dot is
                #     | *
                # ------
                # 0 | 1
                # 1 | 0
                transitions = ((0, self.domain.one()),)

            elif kind == "merge":
                # Localization matrix of merge is
                #     | 00 | 10 | 01 | 11
                # ----------------------
                # 0 |  1 |  0 |  0 |  1
                # 1 |  0 |  1 |  1 |  0
                if local == 0: # 00
                    transitions = ((0, self.domain.one()),)
                elif local == 1: # 10
                    transitions = ((1, self.domain.one()), )
                elif local == 2: # 01
                    transitions = ((1, self.domain.one()), )
                elif local == 3: # 11
                    transitions = ((0, self.domain.one()), )

            elif kind == "split":
                color = word[i]
                alpha = self.root(actual_prefix, color)
                # Localization matrix of split is
                #     |     0    |     1
                # -------------------------
                # 00 |  1/alpha |     0
                # 10 |     0    | -1/alpha
                # 01 |     0    |  1/alpha
                # 11 | -1/alpha |     0
                if local == 0: 
                    transitions = ((0, 1/alpha), (3, -1/alpha))
                elif local == 1:
                    transitions = ((2, 1/alpha), (1, -1/alpha))

            elif kind == "cap":
                color = word[i]
                alpha = self.root(actual_prefix, color)
                # Localization matrix of cap is
                #     |   00  | 10 | 01 |  11
                # ----------------------------
                # * | alpha |  0 |  0 | alpha
                if local == 0: #00
                    transitions = ((0, alpha), )
                elif local == 1: #10
                    transitions = ()
                elif local == 2: #01
                    transitions = ()
                elif local == 3: #11
                    transitions = ((0, alpha), )

            elif kind == "cup":
                color = operation.color
                alpha = self.root(actual_prefix, color)
                # Localization matrix of cup is
                #     |     *
                # ---------------
                # 00 |  1/alpha
                # 10 |     0
                # 01 |     0
                # 11 | -1/alpha
                transitions = ((0, 1/alpha), (3, -1/alpha))

            for new_local, coefficient in transitions:
                new_mask = low + (new_local << i) + (high << (i+inserted))
                output[new_mask] += coefficient*value
        return target, {mask: value for mask, value in output.items() if value}

    def column(self, morphism, column, prefix=None):
        """
            We start with a morphism `morphism` and a sparse vector `column` 
            which is stored as a dictionary whose keys are subexpressions (stored as masks) of the domain of `morphism`.
            The result is the application of the localization matrix for `column` applied to `column`.
            This is also a sparse column whose keys are subexpressions of the codomain of `morphism`.

            We do not compute the localization matrix of `localization`. Instead, we decompose `morphism` into
            a series of operations in which the localization matrix is easy to apply. The localization matrices
            for these operations are then applied successively to compute the output.

            We also allow a `prefix` to be included which amounts to apply `prefix` to the localization matrix.
            This is required to compute the tensor product of localization matrices. When `prefix` is not set, it defaults to the identity element.

            The actual implementation of the column computation is in :func:`column`. This method is simply a wrapper that validates that the input data is correct.
            
        """
        require(morphism.system is self.system, "Evaluator and morphism use different Coxeter systems.")
        column = check_integer(column, "column mask")
        require(0 <= column < 1 << len(morphism.source), f"Column mask is out of range. The column mask provided was {column}, but valid masks satisfy 0 <= mask < {1 << len(morphism.source)}")
        prefix = self.system.identity if prefix is None else prefix
        return self._column(morphism, column, prefix)

    def entry(self, morphism, row, column, prefix=None):
        """Return one entry at integer (target row, source column) masks in the localization matrix of morphism.

        The morphism for computing the localization matrix depends on the kind of morphism.
            - double leaf: if self.leaf_method='weighted' this uses the weighted dot product formula developed in the companion paper; if self.leaf_method='Elias-Williamson graphs' uses :func:`column`.
            - expansion: defer to `leaves.py`
            - sum: take sum of the corresponding localization entries
            - scale: multiply the corresponding localization entry of the child by the scalar
            - otherwise: finds the corresponding column of the localization matrix using :func:`column` and takes the respective entry
        """
        prefix = self.system.identity if prefix is None else prefix
        row, column = check_integer(row, "row mask"), check_integer(column, "column mask")
        require(morphism.system is self.system and 0 <= row < 1 << len(morphism.target)
                and 0 <= column < 1 << len(morphism.source), "Invalid localization entry.")
        if morphism.kind == "double_leaf" and self.leaf_method == "weighted":
            from leaves import double_leaf_entry
            return double_leaf_entry(self, morphism.source, morphism.target, *morphism.payload, row, column, prefix)
        if morphism.kind == "expansion":
            return self._expansion_index(morphism.payload).entry(row, column, prefix)
        if morphism.kind == "sum":
            return sum((self.entry(child, row, column, prefix) for child in morphism.children), self.domain.zero())
        if morphism.kind == "scale":
            return self.polynomial(morphism.payload, prefix)*self.entry(morphism.children[0], row, column, prefix)
        return self.column(morphism, column, prefix).get(row, self.domain.zero())

    def _column(self, morphism, column, prefix):
        """
         We start with a morphism `morphism` and a sparse vector `column` 
        which is stored as a dictionary whose keys are subexpressions (stored as masks) of the domain of `morphism`.
        The result is the application of the localization matrix for `column` applied to `column`.
        This is also a sparse column whose keys are subexpressions of the codomain of `morphism`.

        We do not compute the localization matrix of `localization`. Instead, we decompose `morphism` into
        a series of operations in which the localization matrix is easy to apply. The localization matrices
        for these operations are then applied successively to compute the output.

        We also allow a `prefix` to be included which amounts to apply `prefix` to the localization matrix.
        This is required to compute the tensor product of localization matrices. When `prefix` is not set, it defaults to the identity element.

        Case-by-case in terms of morphism.kind,
            - zero: If `morphism` is 0, then Λ(0)c= 0
            - sum: If `morphism` is of the form F+G, then , then Λ (F+G) c = Λ (F) c + Λ (G) 
            - scale: If `morphism` is of the form μF, then Λ (μF) c = μ Λ (F) c
            - compose: If `morphism` is of the form G o F, then Λ (G o F) c = Λ (G) Λ (F) c
            - tensor: If `morphism` is the form F ⊗ G, then we can write c = c_1 c_2 for c_1 a subxpression mask for the domain of F and c_2 a subexpression mask for the domain of G.
                        Twist the right coefficients by prefix times the left source endpoint.
            - double_leaf: defer to :func:`_double_leaf`. This in turn depends on `self.leaf_method`.
            - expansion: defer to :func:`_expansion_index`

          Method uses LRU cache to speed up repeated computations.
        """
        kind = morphism.kind
        output = defaultdict(self.domain.zero)
        if kind == "zero":
            return MappingProxyType({})
        if kind == "Elias-Williamson graph":
            word, vector = morphism.source, {column: self.domain.one()}
            for op in morphism.payload:
                word, vector = self.apply_operation(word, op, vector, prefix)
            return MappingProxyType(vector)
        if kind == "sum":
            # If `morphism` is of the form F+G, then Λ (F+G) c = Λ (F) c + Λ (G) c
            for child in morphism.children:
                for row, value in self.column(child, column, prefix).items():
                    output[row] += value
        elif kind == "scale":
            # If `morphism` is of the form μF, then Λ (μF) c = μ Λ (F) c
            scalar = self.polynomial(morphism.payload, prefix)
            output.update({row: scalar*value for row, value in self.column(morphism.children[0], column, prefix).items()})
        elif kind == "compose":
            # If `morphism` is of the form G o F, then Λ (G o F) c = Λ (G) Λ (F) c
            first, second = morphism.children
            for middle, low in self.column(first, column, prefix).items():
                for row, top in self.column(second, middle, prefix).items():
                    output[row] += top*low
        elif kind == "tensor":
            # If `morphism` is the form F ⊗ G, then we can write c = c_1 c_2 for c_1 a subxpression mask for the domain of F and c_2 a subexpression mask for the domain of G.
            # Twist the right coefficients by prefix times the left source endpoint. 
            left, right = morphism.children
            left_mask = column & ((1 << len(left.source))-1)
            right_mask = column >> len(left.source)
            right_prefix = self.system.multiply(prefix, self.system.subexpression(left.source, left_mask).endpoint)
            for a, x in self.column(left, left_mask, prefix).items():
                for b, y in self.column(right, right_mask, right_prefix).items():
                    output[a+(b << len(left.target))] += x*y
        elif kind == "double_leaf":
            double_leaf = self._double_leaf(morphism.source, morphism.target, *morphism.payload)
            return self.column(double_leaf, column, prefix)
        elif kind == "expansion":
            return self._expansion_index(morphism.payload).column(column, prefix)
        else:
            raise ValueError("Unknown morphism node.")
        return MappingProxyType({row: value for row, value in output.items() if value})

    def _double_leaf(self, source, target, ea, eb):
        """Construct the double leaf indexed by ea (source) and eb (target). This is given by LL(target, eb)^dual @ LL(source, ea).

        Import the optional light-leaf constructor lazily, build the two light leaves as Elias-Williamson graphs, and
        compose them. The leaves must have matching intermediate endpoints.

          Method uses LRU cache to speed up repeated computations.
        """
        from leaves import light_leaf
        return light_leaf(self.system, source, ea).then(light_leaf(self.system, target, eb).dual())

    def _expansion_index(self, expansion):
        """Build the optional ExpansionEvaluation associated to an expansion.

          Method uses LRU cache to speed up repeated computations.
        """
        from leaves import ExpansionEvaluation
        return ExpansionEvaluation(self, expansion)

    def localization_matrix(self, morphism):
        """Return all nonzero entries of the localization matrix as a read-only {(row, column): value} mapping.

        Evaluate every source mask column and collect its entries. This is a mapping,
        not a Sage matrix; an empty mapping can represent any zero-map shape, whose
        boundaries remain on the morphism. There are 2**len(source) columns and up to
        2**(len(source)+len(target)) entries, so use column/entry for large examples.

        The localization matrix is built column-by-column using :func:`column`
        """
        loc_matrix = dict()
        for col_mask in range(1 << len(morphism.source)):
            column = self.column(morphism, col_mask)
            for row_mask, entry in column.items():
                loc_matrix[(row_mask, col_mask)] = entry
        return MappingProxyType(loc_matrix)

    def print_localization_matrix(self, morphism):
        """Prints each nonzero entry of the localization matrix; return None.

        Convert row and column masks to subexpressions. 
        This inherits the exponential cost of ``localization_matrix``.
        Should only be used for small localization matrices.
        """
        loc_matrix = self.localization_matrix(morphism)
        source_length = len(morphism.source)
        target_length = len(morphism.target)
        for (r, c), value in loc_matrix.items():
            r_print = "".join(map(str, mask_to_subexpression(r, target_length)))
            c_print = "".join(map(str, mask_to_subexpression(c, source_length)))
            print(f"({r_print}, {c_print}) : {value}")
