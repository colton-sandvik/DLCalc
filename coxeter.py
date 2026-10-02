"""Sage-backed Coxeter systems and the combinatorics of subexpressions.

Sage supplies group multiplication, equality, length, Bruhat order, and reduced
expressions. This module adds the conventions needed to specify a double-leaves
basis: masks, defects, a deterministic reduced word, and deterministic rex paths.
It never enumerates the ambient Coxeter group. Infinite groups are allowed.

There are named presets A3, B3, H3 using the golden realization. 
We use the same convention for simple reflections as the accompanying paper.
In particular, H3 has its 5-edge between s and t. Sage's own
named H3 uses a different numbering; ``from_sage`` preserves Sage's numbering.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
from math import comb
from numbers import Integral

from sage.all import AA, QQ, QQbar, ZZ, Graph, CoxeterGroup, CoxeterMatrix, PolynomialRing, identity_matrix, matrix, vector

from formats import check_integer, sha256, require
from quadratic import decode_scalar, encode_scalar, field_data, field_from_data, golden_field, phi

Word = tuple


@dataclass(frozen=True)
class Subexpression:
    """
    A data class for handling subexpressions of expressions in W
    """

    mask: int # The acutal subexpression (e_1, ..., e_k) encoded as an integer, e.g., for k=3, 6 -> (0,1,1)
    endpoint: object # The element in W which evaluates the subexpression, i.e., s_1^e_1 ... s_k^e_k
    defect: int # the defect of the subexpression = #U0 - #D0
    bruhat_stroll: tuple # the Bruhat stroll of the expression i.e., (s_1^e_1, s_1^e_1 s_2^e_2, ..., s_1^e_1 ... s_k^e_k)
    decorations: tuple[str, ...] # the sequence of decorations (dec_1 (e), ..., dec_k (e)) where dec_i (e) is either U0, U1, D0, D1

def monomials(degree: int, rank: int) -> tuple[tuple[int, ...], ...]:
    """
    Produces all possible monomials on ``rank`` many variables of degree ``degree``. 
    The output consists of a tuple of ``rank``-tuples where the sum of the entries are ``degree``. This tuple is listed in lexicographic order.

    Example: monomials(2,3) = ( (0,0,2), (0,1,1), (0,2,0), (1,0,1), (1,1,0), (2,0,0) ) 
    These correspond to the monomials ( z^2, yz, y^2, xz, xy, x^2)
    """
    degree, rank = check_integer(degree, "degree"), check_integer(rank, "rank")
    if degree < 0 or rank < 1:
        raise ValueError("A nonnegative degree and positive rank are required.")
    if rank == 1:
        return ((degree,),)
    return tuple((i,) + tail for i in range(degree+1)
                 for tail in monomials(degree-i, rank-1))


def polynomial_dimension(degree: int, rank: int) -> int:
    """
    Computes the dimension of the vector space of degree ``degree`` polynomials in ``rank`` many variables.
    This is computed by the binomial coefficient binom(rank + degree - 1, rank - 1).

    Example: ``polynomial_dimension(2,3) = 6``
    For the monomial basis of this vector space, see ``monomials``.
    """
    degree, rank = check_integer(degree, "degree"), check_integer(rank, "rank")
    if rank < 1:
        raise ValueError("A positive rank is required.")
    return 0 if degree < 0 else comb(degree + rank - 1, rank - 1)

def subexpression_to_mask(e):
    """
    Converts a tuple e = (e_1, ..., e_k) to an integer 'mask' which is used for most of our computations.
    This is computed by
        mask (e_1, ..., e_n) = e_1 2^0 + e_2 2^1 + ... e_k 2^(k-1)

    The result is `mask` and the length `k` of the subexpression
    """
    mask = 0
    for i in range(len(e)):
        if e[i] != 0 and e[i] != 1:
            raise ValueError("The entries of a subexpression must be either 0 or 1.")
        mask += e[i] * (1 << i)
    return mask, len(e)

def mask_to_subexpression(mask, length):
    """
    Converts a mask to a subexpression (e_1, ..., e_length).
    This extracts the binary representation of the integer `mask`, read from the least significant bit to the most significant bit, and returns a sequence of length `length`
    """
    mask = check_integer(mask, "mask")
    length = check_integer(length, "length")
    if length < 0:
        raise ValueError("The length must be nonnegative.")
    if not 0 <= mask < 1 << length:
        raise ValueError("The mask must be between 0 and 2^length - 1.")
    return tuple((mask >> i) & 1 for i in range(length))



class CoxeterSystem:
    """A Sage Coxeter group together with a balanced realization.

    Parameters
    ----------
    data:
        ``"A3"``, ``"B3"``, ``"H3"``, a square Coxeter matrix, a Sage Cartan
        type, or an existing Sage Coxeter group. Use ``None`` or ``-1`` for an
        infinite off-diagonal label in a matrix.
    labels:
        Distinct strings or integers used in words. Their order fixes all
        lexicographic choices. Defaults to (s,t,u) for the three presets and
        otherwise to Sage's index set.
    cartan:
        Optional Cartan matrix with entries <alpha_i^vee,alpha_j>. If omitted,
        use the preset or the symmetric geometric realization.
    field:
        QQ, ``golden_field()``, or AA.

    Attributes ``group``, ``field``, ``ring``, and ``cartan`` are Sage objects.
    Simple-root variables, of degree 2, are ``ring.gens()``. Their ordinary
    polynomial degree is used when discussing interpolation.
    """

    def __init__(self, data="H3", *, labels=None, cartan=None, field=None):
        preset = data if isinstance(data, str) and data in ("A3", "B3", "H3") else None
        supplied_group = data if hasattr(data, "simple_reflections") and hasattr(data, "coxeter_matrix") else None
        if preset:
            m = {"A3": 3, "B3": 4, "H3": 5}[preset]
            orders = [[1, m, 2], [m, 1, 3], [2, 3, 1]]
            labels = tuple(labels) if labels is not None else ("s", "t", "u")
            if cartan is None:
                x, y = {"A3": (-1, -1), "B3": (-2, -1), "H3": (-phi(), -phi())}[preset]
                cartan = [[2, x, 0], [y, 2, -1], [0, -1, 2]]
        elif supplied_group is not None:
            indices = tuple(supplied_group.index_set())
            cm = supplied_group.coxeter_matrix()
            orders = [[cm[i, j] for j in indices] for i in indices]
            labels = tuple(labels) if labels is not None else indices
        else:
            if isinstance(data, (list, tuple)) and data and isinstance(data[0], (list, tuple)):
                data = [[-1 if entry is None else entry for entry in row] for row in data]
            cm = CoxeterMatrix(data)
            indices = tuple(cm.index_set())
            orders = [[cm[i, j] for j in indices] for i in indices]
            labels = tuple(labels) if labels is not None else indices
        self.rank = len(orders)
        require(self.rank > 0, "The coefficient ring must have positive finite rank.")
        self.labels = tuple(int(i) if isinstance(i, Integral) and not isinstance(i, bool) else i for i in labels)
        require(len(self.labels) == self.rank and len(set(self.labels)) == self.rank, "Invalid generator labels.")
        require(all(isinstance(i, (str, int)) and not isinstance(i, bool) for i in self.labels),
                "Portable generator labels must be strings or Python integers.")
        self._positions = {label: i for i, label in enumerate(self.labels)}

        def order(value):
            if value is None or value == -1 or str(value) in ("+Infinity", "Infinity"):
                return None
            return int(value)

        self.orders = tuple(tuple(order(v) for v in row) for row in orders)
        require(all(len(row) == self.rank for row in self.orders), "Coxeter matrix must be square.")
        for i in range(self.rank):
            require(self.orders[i][i] == 1, "Coxeter diagonal must be 1.")
            for j in range(i):
                m = self.orders[i][j]
                require(m == self.orders[j][i] and (m is None or m >= 2), "Invalid Coxeter matrix.")
        if cartan is None:
            cartan = [[2 if i == j else -self._geometric_pairing(self.orders[i][j])
                       for j in range(self.rank)] for i in range(self.rank)]
        if field is None:
            for candidate in (QQ, golden_field(), AA):
                try:
                    matrix(candidate, cartan)
                    field = candidate
                    break
                except (TypeError, ValueError):
                    pass
        field_data(field)  # Check that the field has a supported exact codec.
        self.field = field # The field of definition for the realization, either QQ, QQ (phi), or AA
        self.cartan = matrix(field, cartan) # The cartan matrix of the realization
        require(self.cartan.dimensions() == (self.rank, self.rank), "Cartan dimensions disagree with the Coxeter matrix.") # We only support realizations in which the dimension of the realization is the same as the rank of the Coxeter group
        self.cartan.set_immutable()
        self._check_realization()
        names = tuple("alpha_" + str(i) for i in range(self.rank))
        self.ring = PolynomialRing(field, self.rank, names=names, order="degrevlex") # The polynomial ring R=Sym (V) where V is the realization. This is a polynomial ring over ``self.field`` with ``self.rank`` many variables
        if supplied_group is None:
            raw = [[-1 if m is None else m for m in row] for row in self.orders]
            self._sage_indices = tuple(range(self.rank))
            self.group = CoxeterGroup(CoxeterMatrix(raw, index_set=self._sage_indices), base_ring=AA) # The Coxeter group
        else:
            self.group = supplied_group # The Coxeter group
            self._sage_indices = tuple(self.group.index_set())
        self._sage_label = dict(zip(self.labels, self._sage_indices))
        self._public_label = dict(zip(self._sage_indices, self.labels))
        self.simple = {i: self.group.simple_reflection(self._sage_label[i]) for i in self.labels}  # A dictionary of the simple reflections in the Coxeter group. The keys are the labels ``self.labels`` and the values are the simple reflections in W=``self.group``. 
        self.identity = self.group.one() # Identity element in W = ``self.group``

        """
        The collection of reflection matrices. This is a dictionary whose keys are labels of simple reflections (``self.labels``).
        The value at a simple reflection ``s`` is the ``self.rank``x``self.rank`` square matrix which encodes the action of s on the realization V
        """
        self._reflection_matrices = {} # The collection of reflection matrices. 
        for label, i in self._positions.items():
            reflection = identity_matrix(field, self.rank)
            for j in range(self.rank):
                reflection[i, j] -= self.cartan[i, j]
            reflection.set_immutable()
            self._reflection_matrices[label] = reflection

        self._descriptor = self.to_data()
        self.fingerprint = sha256(self._descriptor)

    @staticmethod
    def _geometric_pairing(m):
        """
        Computes the value of the (s,t) or (t,s) entry of the Cartan matrix for the "golden realization". 
        The input ``m`` is the order of st. 
        """
        if m is None:
            return QQ(2)
        if m == 2:
            return QQ(0)
        if m == 3:
            return QQ(1)
        if m == 5:
            return phi()
        return AA(2*QQbar.zeta(2*m).real())

    @classmethod
    def from_sage(cls, group, **kwargs):
        """Use an existing Sage group, retaining its elements and index order."""
        return cls(group, **kwargs)

    @classmethod
    def dihedral_product(cls, m: int):
        """
        This produces a CoxeterSystem for the group I2(m) x A1 for any m >=2.

        The ambient realization has Cartan matrix
                [ 2 x 0 ]
                [ y 2 0 ]
                [ 0 0 2 ]
        where 
            m=2 -> x=0, y=0
            m=3 -> x=-1, y=-1
            m=4 -> x=-2, y=-1
            m=5 -> x=-phi, y=-phi
            m=6 -> x=-3, y=-1
            m > 6 -> x=y= -2 * (a primitive 2m-th root of unity)

        The ground field of this realization is
            m=2,3,4,6 - QQ
            m=5 - QQ(phi)
            m>6 - AA
        """
        m = check_integer(m, "m")
        if m < 2:
            raise ValueError("m must be at least 2.")
        x, y = {2: (0, 0), 3: (-1, -1), 4: (-2, -1),
                5: (-phi(), -phi()), 6: (-3, -1)}.get(m, (None, None))
        if x is None:
            x = y = -cls._geometric_pairing(m)
        return cls([[1, m, 2], [m, 1, 2], [2, 2, 1]], labels=("s", "t", "u"),
                   cartan=[[2, x, 0], [y, 2, 0], [0, 0, 2]])

    def _check_realization(self):
        """Check balance and all finite-edge quantum conditions exactly."""
        C = self.cartan
        for i in range(self.rank):
            require(C[i, i] == 2, "Cartan diagonal must be 2.") # Checks that the diagonal entries of the Cartan matrix are all 2
            for j in range(i):
                x, y, m = -C[i, j], -C[j, i], self.orders[i][j]
                require(AA(x) >= 0 and AA(y) >= 0, "Use the positive geometric embedding of the realization.") # Checks that the off-diagonal entries of the Cartan matrix are <= 0
                if m is None:
                    require(AA(x*y) >= 4, "An infinite edge requires Cartan product at least 4.") 
                    continue
                qs, qt = [self.field(0), self.field(1)], [self.field(0), self.field(1)]
                for n in range(2, m+1):
                    qs.append(x*qt[n-1]-qs[n-2])
                    qt.append(y*qs[n-1]-qt[n-2])
                require(qs[m] == qt[m] == 0, "Finite-edge quantum numbers do not vanish.") # Checks that the top most two-colored quantum numbers vanish
                require(qs[m-1] == qt[m-1] == 1, "This computational interface requires a balanced realization.") # Checks that the realization is balanced
                require(all(qs[n] and qt[n] for n in range(1, m)), "A required rank-two quantum denominator vanishes.")

    def word(self, word) -> Word:
        """
        The input is a string ``word`` which is broken up into a tuple where each entry labels a simple reflection.

        Example: ``word ('stsut') = ('s', 't', 's', 'u', 't')``.

        Throws an error if the word contains an unknown label for a simple reflection
        For example, if S = {s,t}, then ``word('stu')`` would throw an error.  
        """
        if isinstance(word, str):
            if word == "" or word == "e" and "e" not in self.labels:
                return ()
            require(all(isinstance(i, str) and len(i) == 1 for i in self.labels),
                    "Use a tuple of labels for this Coxeter system.")
        result = tuple(word)
        require(all(i in self._positions for i in result), "A word contains an unknown generator.")
        return tuple(self.labels[self._positions[i]] for i in result)

    def word_key(self, word) -> tuple[int, ...]:
        """
        The input is either a string ``word`` or a tuple of simple reflection labels. 
        The output is a tuple where each entry is the index of the simple reflection in ``self.reflections``.

        Example: for ``s -> 0, t-> 1, u-> 2``
            1) ``word ('stsut') = (0,1,0,2,1)``
            2) ``word (('s', 't', 'u', 's')) = (0,1,2,0)``
        """
        return tuple(self._positions[i] for i in word)

    @lru_cache(maxsize=None)
    def multiply_simple(self, w, label):
        """Cached right multiplication, performed by Sage."""
        return w*self.simple[label]

    @lru_cache(maxsize=None)
    def multiply(self, left, right):
        """Sage multiplication, avoiding repeated identity and prefix products."""
        if left == self.identity:
            return right
        if right == self.identity:
            return left
        return left*right

    @lru_cache(maxsize=None)
    def length(self, w) -> int:
        """Sage's Coxeter length of an element."""
        return int(w.length())

    @lru_cache(maxsize=None)
    def bruhat_le(self, x, y) -> bool:
        """Sage's Bruhat comparison, cached for repeated dominance tests."""
        return bool(x.bruhat_le(y))

    def element(self, word):
        """Return the Sage element represented by a public-label word."""
        w = self.identity
        for label in self.word(word):
            w = self.multiply_simple(w, label)
        return w

    @lru_cache(maxsize=None)
    def reduced_word(self, w) -> Word:
        """Lexicographically least reduced word via Sage left descents.

        Choosing the first available left descent recursively gives the
        lexicographically least reduced expression. This convention is
        independent of the reduced word selected internally by Sage.
        """
        if w == self.identity:
            return ()
        for label in self.labels:
            if w.has_descent(self._sage_label[label], side="left"):
                return (label,) + self.reduced_word(self.simple[label]*w)
        raise ArithmeticError("Sage returned a nonidentity element without a descent.")

    def braid_order(self, s, t):
        """Return m_st. When m_st = infty, the method returns None"""
        return self.orders[self._positions[s]][self._positions[t]]

    def braid_neighbors(self, word):
        """
        Starting with a ``word``, we look for all possible words obtained from ``word`` by applying a single braid move, i.e., where sts... is replaced by tst...

        The output is a generator that produces tuples (``new_word``, ``position``, ``length``). 
        The ``new_word`` is the word obtained from ``word`` by applying a braid move at position ``position`` for simple reflections s and t with m_st = ``length``

        Example: for m_st = 5, m_tu = 3, and m_su = 2
            ``list(self.braid_neighbors('ststsu')) = (
                (('t', 's', 't', 's', 't', 'u'), 0, 5),
                (('s', 't', 's', 't', 'u', 's'), 4, 2)
            )
        """
        word = self.word(word)
        for position in range(len(word)-1):
            s, t = word[position:position+2]
            m = self.braid_order(s, t)
            if s == t or m is None or position+m > len(word):
                continue
            if word[position:position+m] == tuple(s if j % 2 == 0 else t for j in range(m)):
                replacement = tuple(t if j % 2 == 0 else s for j in range(m))
                yield word[:position]+replacement+word[position+m:], position, m

    @lru_cache(maxsize=None)
    def rex_path(self, source: Word, target: Word) -> tuple:
        """Deterministic breadth-first rex path between reduced expressions.

        Each step is (position, braid_length). The search only traverses the
        finite set of reduced expressions of this one element, even when the
        ambient group is infinite. Long reduced-expression graphs can still
        be expensive; no claim of polynomial-time path finding is made.

        Example: for m_st = 3, m_tu = 3, and m_su = 2
        The longest element has reduced expressions 'stusts' and 'utustu'
        `` self.rex_path('stusts', 'utustu') =  (
            (2,2),
            (0,3),
            (2,3),
            (1,2),
            (4,2),
            (2,3),
            (0,3)
        )
        This corresponds to the path
        stusts -> stsuts -> tstuts -> tsutus -> tustus -> tustsu -> tutstu -> utustu
        """
        source, target = self.word(source), self.word(target)
        endpoint = self.element(source)
        require(endpoint == self.element(target) and len(source) == len(target) == self.length(endpoint),
                f"A rex path requires reduced expressions of the same element. Current source is {source} and current target is {target}")
        if source == target:
            return ()
        parents = {source: None}
        queue = deque([source])
        while queue:
            current = queue.popleft()
            for next_word, position, m in self.braid_neighbors(current):
                if next_word in parents:
                    continue
                parents[next_word] = (current, (position, m))
                if next_word == target:
                    path = []
                    while parents[next_word] is not None:
                        next_word, step = parents[next_word]
                        path.append(step)
                    return tuple(reversed(path))
                queue.append(next_word)
        raise ArithmeticError("Reduced expressions were not connected by braid moves. This is very bad since the rex graph is connected. Something must have gone wrong.")

    @lru_cache(maxsize=None)
    def subexpression(self, word: Word, mask: int) -> Subexpression:
        """
        Given a word ``word`` and an integer ``mask`` with 0 <= mask < 2^len(word), produce a ``Subexpression``.

        Example: let m_st = 3 
        ``self.subexpression('sts', 5) = Subexpression(
            endpoint = [ 1 0 0 ]
                       [ 0 1 0 ]
                       [ 0 0 1 ],
            defect = 1,
            bruhat_stroll = (
                [ -1 1 0 ]   [ -1 1 0 ]   [ 1 0 0 ]
                [  0 1 0 ]   [  0 1 0 ]   [ 0 1 0 ]
                [  0 0 1 ],  [  0 0 1 ],  [ 0 0 1 ]
            )
            decoration = ('U1', 'U0', 'D1')
        )``

        In this case, the subexpression is (1, 0, 1) for the word sts. The Bruhat stroll is (s, s, e). 
        """
        word = self.word(word)
        mask = check_integer(mask, "subexpression mask")
        require(0 <= mask < 1 << len(word), "Mask is out of range.")
        w, defect, bruhat_stroll, decorations = self.identity, 0, [], []
        for position, label in enumerate(word):
            ws = self.multiply_simple(w, label)
            up = self.length(ws) > self.length(w)
            selected = (mask >> position) & 1
            decorations.append(("U" if up else "D") + str(selected))
            if selected:
                w = ws
            else:
                defect += 1 if up else -1
            bruhat_stroll.append(w)
        return Subexpression(mask, w, defect, tuple(bruhat_stroll), tuple(decorations))

    @lru_cache(maxsize=None)
    def subexpressions(self, word: Word) -> tuple[Subexpression, ...]:
        """
        Produces a tuple of all possible subexpressions for the word ``word``. 
        """
        word = self.word(word)
        return tuple(self.subexpression(word, mask) for mask in range(1 << len(word)))

    def defect_counts(self, word) -> Counter:
        """
        Produces a dictionary (or more aptly a Counter) that records the frequencies of a pair (``w``, ``defect``) appearing in the list of subexpressions.

        To illustrate this, we can consider the analagous concept in math. Let s be a simple reflection. Then 'sss' has 8 subexpressions
        000 -> defect = 3  and w = e 
        100 -> defect = -2 and w = s 
        010 -> defect = 0  and w = s 
        110 -> defect = 1  and w = e
        001 -> defect = 2  and w = s
        101 -> defect = -1 and w = e 
        011 -> defect = 1  and w = e
        111 -> defect = 0  and w = s

        So the output would be something like
        (s, -2) : 1
        (e, -1) : 1
        (s, 0)  : 2
        (e, 1)  : 2
        (s, 2)  : 1
        (e, 3)  : 1
        """

        subexps = self.subexpressions(word)
        w_def_pairs = [] # list of pairs (w, defect) for the subexpressions
        for subexp in subexps: # loop through all subexpressions
            w = subexp.endpoint
            defect = subexp.defect
            w_def_pairs.append((w, defect)) # add (w, defect) to the list of w_def_pairs
        return Counter(w_def_pairs) # counts up the occurrences of each (w, defect) pair

    def hom_profile(self, source, target, degree: int = 0) -> dict:
        """Records the k-dimension of the degree ``degree`` Hom space from B_source to B_target
            Output is a dictionary: {
                "degree": degree,
                "dimension": the total dimension of the degree ``degree`` Hom space,
                "candidates": the number of light leaves that contribute to the dimension,
                "basis_degrees": a dictionary consisting of {d: num} where d is the degree of a collection of double leaves and num is the number of double leaves which have that degree
                                the total of the values in this dict is the number of candidates 
            }

        Example: W = H_3 with the golden realization. Then
        ``W.hom_profile("utstuststuststs","ststsutstsutstu", 0) = {
            'degree': 0,
            'dimension': 94915,
            'candidates': 78417,
            'basis_degrees': {
                -8: 1,
                -6: 34,
                -4: 578,
                -2: 6644,
                0: 71160
            }
        }

        The graded Hom space Hom^* (B_source, B_target) decomposes as a direct sum ⊕_w ⊕_{e.endpoint = w = f.endpoint} R LL_{e,f}^w
        The degree of LL_{e,f}^w is e.defect + f.defect, so we have that Hom^(degree) (B_source, B_target) = ⊕_w ⊕_{e.endpoint = w = f.endpoint} R^(degree - e.defect - f.defect ) LL_{e,f}^w
        This gives the following algorithm for computing the dimension of Hom^(degree) (B_source, B_target).
        1) List all the subexpressions for target. We only need to record the triples (w, defect, count) which details
            how many subexpressions (count) of target evaluate to w and have a given defect. For data handling purposes, we record this as a key-value pair w : (defect, count)
        2) List all the subexpressions for source. Again, we only need to record the triples (w, defect, count).
        3) For each triple (w, defect, count) for the source, we loop over all (defect', count') labeled by w for the target.
        4) The contribution of the (w, defect, count) and (defect', count') to the degree ``degree`` Hom space is dim (R^(degree - defect - defect')) x count x count'. 
            The graded dimension of R can be computed using general facts about homogenous polynomials 
        """
        degree = check_integer(degree, "morphism degree")
        left = self.defect_counts(source)
        right = self.defect_counts(target)
        by_endpoint = defaultdict(list)
        # Record all triples (w, defect, count) for the right-hand side (step 1)
        for (w, right_defect), right_count in right.items():
            by_endpoint[w].append((right_defect, right_count)) 

        total = Counter()
        dimension = 0
        # Loop over all triples (w, defect, count) for the left-hand side (step 2)
        for (w, left_defect), left_count in left.items():
            # Only consider the subexpressions which have the same endpoint (step 3)
            for right_defect, right_count in by_endpoint[w]:
                double_leaf_degree = left_defect + right_defect # degree of the double leaf = sum of defects for source subexpression and target subexpression
                if degree >= double_leaf_degree and (degree-double_leaf_degree) % 2 == 0:
                    total[double_leaf_degree] += left_count*right_count # records the number of double leaves appearing; not the dimension of the Hom space
                    dimension += left_count*right_count*polynomial_dimension((degree-double_leaf_degree)//2, self.rank) # add to the running dimension total (step 4)
        return {"degree": degree, "dimension": dimension, "candidates": sum(total.values()),
                "basis_degrees": dict(sorted(total.items()))}

    @lru_cache(maxsize=None)
    def action_matrix(self, w):
        """Matrix of a Sage element on the chosen simple-root realization."""
        result = identity_matrix(self.field, self.rank)
        for label in self.reduced_word(w):
            result = result*self._reflection_matrices[label]
        result.set_immutable()
        return result

    def act(self, w, polynomial):
        """
        Apply w to a polynomial in the simple-root variables (i.e. an element of R).
        This is computed from the Cartan matrix for the realization.
        """
        polynomial = self.ring(polynomial)
        if w == self.identity:
            return polynomial
        roots = self.ring.gens()
        M = self.action_matrix(w)
        images = [sum(M[i, j]*roots[i] for i in range(self.rank)) for j in range(self.rank)]
        return self.ring(polynomial(*images))

    def to_data(self) -> dict:
        """Produces the basic information needed to reproduce the object.
        Consists of 
        1) the labels for the simple reflections, 
        2) the Coxeter_matrix of the underlying Coxeter group, 
        3) the ground field of the realization
        4) the Cartan matrix for the realization
        """
        return {"labels": list(self.labels), "coxeter_matrix": [list(row) for row in self.orders],
                "field": field_data(self.field),
                "cartan": [[encode_scalar(c, self.field) for c in row] for row in self.cartan.rows()]}

    @classmethod
    def from_data(cls, data: dict):
        """Rebuild and validate a system from its structured descriptor produced by ``self.to_data()``."""
        require(isinstance(data, dict) and set(data) == {"labels", "coxeter_matrix", "field", "cartan"},
                "Invalid system descriptor.")
        field = field_from_data(data["field"])
        cartan = [[decode_scalar(c, field) for c in row] for row in data["cartan"]]
        orders = [[-1 if m is None else m for m in row] for row in data["coxeter_matrix"]]
        return cls(orders, labels=data["labels"], field=field, cartan=cartan)

    def __repr__(self):
        return f"CoxeterSystem(rank={self.rank}, labels={self.labels!r}, field={self.field})"
