"""Fixed light leaves, double leaves, double leaf bases, and lambda functionals.

Every choice is part of the public convention: generator order as declared in
the system, lexicographically least reduced words, and breadth-first rex paths
with braid positions scanned from left to right. These are the same choices
made in the companion paper.

For a source subexpression e and target subexpression f with common endpoint
w, DL(e,f) = dual(LL(f)) @ LL(e). The degree is defect(e)+defect(f). The lambda
functional reads the (f,e) entry of the localization matrix.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

from formats import check_integer, require
from morphisms import Morphism, Operation, mask_to_subexpression


@lru_cache(maxsize=None)
def light_leaf(system, source, mask):    
    """Construct the light leaf morphism with source `source` and subexpression corresponding to `mask`.

    ``source`` is any Bott--Samelson word. At a down label, first rex the current
    reduced prefix to rex(ws)s. D0 uses a merge and D1 a cap. 
    At the end rex to rex(w). 
    U0 uses a upper dot and U1 is an identity. 
    
    The resulting degree is the defect of the subexpression corresponding to mask.
    """
    source = system.word(source)
    subexpression = system.subexpression(source, mask)
    rex = ()
    operations = []
    ue = mask_to_subexpression(mask, len(source))
    decorations = subexpression.decorations
    bruhat_stroll = subexpression.bruhat_stroll
    endpoint = subexpression.endpoint

    for i, s in enumerate(source):
        if decorations[i] == "U1":
            rex = rex + (s,) # Step of the light leaf just adds an identity strand colored by s on the right 
        elif decorations[i] == "U0":
            operations.append(Operation("upper dot", len(rex))) # Step of the light leaf just adds an upper dot colored by s on the right
        elif decorations[i] == "D1":
            rex_s = system.reduced_word(bruhat_stroll[i]) + (s,) # Find a reduced expression from the previous step ending in s. This is deterministic due to our lexicographic ordering on reduced words
            operations.extend(Operation("braid", i, length=m) for i, m in system.rex_path(rex, rex_s)) # Apply braid moves to go from the target of the previous step to the rex move ending in s. This is deterministic due to our fixed choice of rex paths
            operations.append(Operation("cap", len(rex_s) - 1)) # The previous two operations have target of the form ...ss. We apply a cap on these last two strands.
            rex = rex_s[:-1]
        elif decorations[i] == "D0":
            rex_s = system.reduced_word(system.multiply_simple(bruhat_stroll[i], s)) + (s,)  # Find a reduced expression from the previous step ending in s. This is deterministic due to our lexicographic ordering on reduced words
            operations.extend(Operation("braid", i, length=m) for i, m in system.rex_path(rex, rex_s)) # Apply braid moves to go from the target of the previous step to the rex move ending in s. This is deterministic due to our fixed choice of rex paths
            operations.append(Operation("merge", len(rex_s) - 1))# The previous two operations have target of the form ...ss. We apply a merge on these last two strands.
            rex = rex_s
    
    operations.extend(Operation("braid", i, length=m) for i,m in system.rex_path(rex, system.reduced_word(endpoint))) # One last sequence of rex moves to ensure the target is correct
    result = Morphism.elias_williamson_graph(system, source, operations)
    return result

def double_leaf(system, source, target, source_mask, target_mask):
    """
    Constructs a double leaf B_source -> B_target with the given subexpressions
    Does not expand into an Elias-Williamson graph or perform any additional computations.
    Checks that the endpoint of both light leaves matches, before proceeding. 
    To expand into a sequence of operations use :func:`expand_double_leaf` instead.
    """
    source, target = system.word(source), system.word(target)
    e = system.subexpression(source, source_mask)
    f = system.subexpression(target, target_mask)
    require(e.endpoint == f.endpoint, "Double-leaf masks must have the same endpoint.")
    return Morphism(system, source, target, "double_leaf", (e.mask, f.mask))


def expand_double_leaf(value):
    """
    Constructs a double leaf B_source -> B_target with the given subexpressions
    The result is the double leaf as a morphism which has been expanded into a single Elias-Williamson
    graph.
    """
    require(value.kind == "double_leaf", "Expected a double leaf.")
    e, f = value.payload
    return light_leaf(value.system, value.source, e).then(light_leaf(value.system, value.target, f).dual())


def export_leaf(value, path):
    """Save a light leaf or double leaf with the actual ordered elementary operations.

    The output is an ordinary morphism fil. 
    """
    if value.kind == "double_leaf":
        value = expand_double_leaf(value)
    require(value.kind == "Elias-Williamson graph", "Expected a light-leaf or double leaf.")
    return value.save(path)


@dataclass(frozen=True)
class LeafPair:
    """The datum of a double leaf for a fixed source and target. Simply stores the source mask, target mask, the intermediary Coxeter group element, and the degree"""

    source_mask: int
    target_mask: int
    endpoint: object
    degree: int

    @property
    def key(self):
        return self.source_mask, self.target_mask


class DoubleLeavesBasis:
    """The finite part of a double-leaves basis contributing to Hom^degree.

    Endpoint length decreases first. Within an endpoint, a strict increase in
    either subexpression's prefix Bruhat order occurs earlier. This makes the
    matrix lambda_i(D_j) lower triangular: entries with j>i vanish.
    Coefficients of D_i have ordinary polynomial degree (degree-D_i.degree)/2.
    Construction enumerates source and target subexpressions.
    """

    def __init__(self, system, source, target, degree=0):
        self.system = system
        self.source, self.target = system.word(source), system.word(target)
        self.degree = degree = check_integer(degree, "morphism degree")
        by_endpoint = defaultdict(list)
        for f in system.subexpressions(self.target):
            by_endpoint[f.endpoint].append(f)
        entries = []
        for e in system.subexpressions(self.source):
            for f in by_endpoint[e.endpoint]:
                d = e.defect+f.defect
                if d <= degree and (degree-d) % 2 == 0:
                    entries.append(LeafPair(e.mask, f.mask, e.endpoint, d))

        def key(pair):
            e = system.subexpression(self.source, pair.source_mask)
            f = system.subexpression(self.target, pair.target_mask)
            he = sum(system.length(w) for w in e.bruhat_stroll)
            hf = sum(system.length(w) for w in f.bruhat_stroll)
            return (-system.length(pair.endpoint), system.word_key(system.reduced_word(pair.endpoint)),
                    -he-hf, -he, -hf, -pair.source_mask, -pair.target_mask)

        # A sorted tuple of all valid leaf pairs appearing in the double leaves basis
        # This gives the set of double leaves a total ordering needed for the lambda_j (LL_i) matrix to be lower triangular
        self.entries = tuple(sorted(entries, key=key))
        # a tuple of the coefficient degrees using the same sorting as self.entries. The degrees are half the morphism degree, so that they can be used for polynomial arguments
        self.coefficient_degrees = tuple((self.degree-p.degree)//2 for p in self.entries)
        # a tuple of all double leaf morphisms
        self._morphisms = tuple(double_leaf(system, self.source, self.target, *p.key) for p in self.entries)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        return self.entries[index]

    def __repr__(self):
        return f"Double leaf basis for homogeneous morphisms from B_{self.source} to B_{self.target} of degree {self.degree}."

    def morphism(self, index):
        """Return the compact morphism for the indexed double leaf."""
        return self._morphisms[index]

    def lambda_value(self, morphism, index, evaluator):
        """
        Computes the coefficient lambda_(index) (morphism).
        In more mathy notation and terminology, the index (=i) corresponds to a pair (e^i, f^i) where e^i is a subexpression of morphism.source
        and f^i is a subexpression of morphism.target. Then lambda_i (morphism) is the entry (f^i, e^i) of the localization matrix for morphism

        The evaluator is an instance of the `Localization` class. 
        """
        require(morphism.system is self.system and evaluator.system is self.system,
                "Basis, morphism, and evaluator must share a system.")
        require((morphism.source, morphism.target) == (self.source, self.target), "Morphism needs to live in the same morphism space as the double leaves basis.")
        pair = self.entries[index]
        return evaluator.entry(morphism, pair.target_mask, pair.source_mask)

    def diagonal(self, index, evaluator):
        """
        Set i = index
        Computes the coefficient lambda_i (LL_i) using the formula from the companion paper
        
        Equivalently, this is the entry (e^i, f^i) for the localization matrix of LL_i

        The evaluator is an instance of the `Localization` class.
        """
        p = self.entries[index]
        one = self.system.identity
        word = self.system.reduced_word(p.endpoint)
        return (evaluator.zeta(word, (1 << len(word))-1, one)
                / evaluator.zeta(self.target, p.target_mask, one)
                * gamma(evaluator, self.source, p.source_mask, one)
                * gamma(evaluator, self.target, p.target_mask, one))

    def lambda_values(self, morphism, evaluator):
        """Produces the lambda functional in basis order."""
        for i in range(len(self)):
            yield self.lambda_value(morphism, i, evaluator)


def gamma(evaluator, word, mask, prefix=None):
    """ prefix (gamma_e) = prefix * product(U0 beta) * product(D1 -beta)."""
    system = evaluator.system
    prefix = system.identity if prefix is None else prefix
    data = system.subexpression(word, mask)
    w, result = prefix, evaluator.domain.one()
    for i, (color, decoration) in enumerate(zip(word, data.decorations)):
        if decoration == "U0":
            result *= evaluator.root(w, color)
        elif decoration == "D1":
            result *= -evaluator.root(w, color)
        if (mask >> i) & 1:
            w = system.multiply_simple(w, color)
    return result

def double_leaf_entry(evaluator, source, target, e, f, row, column, prefix):
    """Compute one double-leaf entry through the diagonal adjunction form.

    If A=Lambda(LL_e), B=Lambda(LL_f), the reflected matrix is
    diag(zeta_target)^(-1) B^T diag(zeta_red(w)). Thus no complete double-leaf
    matrix is built.
    """
    system = evaluator.system
    a = evaluator.column(light_leaf(system, source, e), column, prefix)
    if not a:
        return evaluator.domain.zero()
    b = evaluator.column(light_leaf(system, target, f), row, prefix)
    middle = system.reduced_word(system.subexpression(source, e).endpoint)
    numerator = sum((value*b.get(h, 0)*evaluator.zeta(middle, h, prefix) for h, value in a.items()),
                    evaluator.domain.zero())
    return numerator/evaluator.zeta(target, row, prefix)


class DominanceIndex:
    """
    This class is used to filter possible contributing double leaves by the path-dominance order.
    The general setup is as follows:
        We want to compute the (f, e)-entry of the localization matrix for each double leaf appearing 
        in the double leaves basis = (DL_1, ... , DL_N) where LL_i = LL_(e_i, f_i).
        However, many of these entries are zero by the path dominance ordering.
        To reduce the number of computations, we instead only consider the i's such that e_i <= e and f_i <= f.
        This filtering is handled by this class. Note no double leaf localization computations are done in this class.

    

    """

    def __init__(self, system, source, target, pairs):
        self.system = system
        self.source = source
        self.target = target
        pairs = tuple(pairs)
        self.num_pairs = len(pairs)
        self.all_bits = (1 << len(pairs))-1
        
        """
        source_positions is a list of dictionaries indexed by (p, w) where 0 <= p < len(source) and w is an element of the Coxeter system
        The value of source_position at (p,w) is a binary number whose i-th digit is 1 if and only if the p-th position 
        of the Bruhat stroll for pair[i].source is w.

        This allows us to conveniently encode all Bruhat strolls for the source subexpressions of the double leaf basis.
        If we want to check that e_p <= e, then we can loop through all w in source_positions[p].keys() and check if
        that if  
        """
        self.source_positions = [defaultdict(int) for _ in self.source]
        for i, pair in enumerate(pairs):
            for p,w in enumerate(system.subexpression(self.source, pair[0]).bruhat_stroll):
                self.source_positions[p][w] |= 1 << i
        
        """
        target_positions is constructed analogously to source_positions with the source being replaced by the target
        """
        self.target_positions = [defaultdict(int) for _ in self.target]
        for i, pair in enumerate(pairs):
            for p,w in enumerate(system.subexpression(self.target, pair[1]).bruhat_stroll):
                self.target_positions[p][w] |= 1 << i

        self._source_upward = lru_cache(maxsize=None)(self._source_upward)
        self._target_upward = lru_cache(maxsize=None)(self._target_upward)
        self._source_contributors = lru_cache(maxsize=None)(self._source_contributors)
        self._target_contributors = lru_cache(maxsize=None)(self._target_contributors)

    def _source_upward(self, position, endpoint):
        """
        Finds all source subexpressions e_i appearing in double leaf pairs such that the `position`-th position
        of the Bruhat stroll of e_i is less than or equal to `endpoint` under the Bruhat order.

        Returns a binary number whose i-th position is 1 if and only if e_i satisfies the above condition. 

        This function has an LRU cache to speed up computations.
        """
        result = 0
        for w, bits in self.source_positions[position].items():
            if self.system.bruhat_le(endpoint, w):
                result |= bits
        return result

    def _target_upward(self, position, endpoint):
        """
        Finds all target subexpressions f_i appearing in double leaf pairs such that the `position`-th position
        of the Bruhat stroll of f_i is less than or equal to `endpoint` under the Bruhat order.

        Returns a binary number whose i-th position is 1 if and only if f_i satisfies the above condition. 

        This function has an LRU cache to speed up computations.
        """
        result = 0
        for w, bits in self.target_positions[position].items():
            if self.system.bruhat_le(endpoint, w):
                result |= bits
        return result

    def _source_contributors(self, mask):
        """
        Returns a binary number where the i-th bit is 1 if and only if the source subexpression indexed by i in the list
        of double leaf pairs is <= to mask in the path dominance order.

        The algorithm starts with saying that all source subexpressions of the double leaf pairs are allowed.
        We then enumerate through each 0<=position<len(source) and then remove all source subexpressions e_i of
        the double leaf pairs such that the `position`-th position of the Bruhat stroll for e_i is not below
        the `position`-th position of the Bruhat stroll for the subexpression e corresponding to `mask`.
        In the end, all that remains is the source subexpressions which are below e in the path dominance
        order. 

        The function returns a binary number whose i-th position is 1 if and only if e_i <= e in the 
        path dominance order.

        LRU cache is implemented to speed up computations
        """
        bits = self.all_bits
        bruhat_stroll = self.system.subexpression(self.source, mask).bruhat_stroll
        for position, endpoint in enumerate(bruhat_stroll):
            bits &= self._source_upward(position, endpoint) 
            if not bits:
                break
        return bits

    def _target_contributors(self, mask):
        """
        Returns a binary number where the i-th bit is 1 if and only if the target subexpression indexed by i in the list
        of double leaf pairs is <= to mask in the path dominance order.

        The algorithm starts with saying that all target subexpressions of the double leaf pairs are allowed.
        We then enumerate through each 0<=position<len(source) and then remove all target subexpressions f_i of
        the double leaf pairs such that the `position`-th position of the Bruhat stroll for f_i is not below
        the `position`-th position of the Bruhat stroll for the subexpression f corresponding to `mask`.
        In the end, all that remains is the source subexpressions which are below f in the path dominance
        order. 

        The function returns a binary number whose i-th position is 1 if and only if f_i <= f in the 
        path dominance order.

        LRU cache is implemented to speed up computations
        """
        bits = self.all_bits
        bruhat_stroll = self.system.subexpression(self.target, mask).bruhat_stroll
        for position, endpoint in enumerate(bruhat_stroll):
            bits &= self._target_upward(position, endpoint)
            if not bits:
                break
        return bits

    def contributors(self, source_mask, target_mask=None):
        """
        Returns the positions of the double leaf pairs which are below (source_mask, target_mask).
        In other words, we find all pairs (e_i, f_i) appearing the double leaves basis such that e_i <= source_mask
        and f_i <= target_mask in the path dominance order.

        If target_mask=None, then we only filter based on the source_mask.

        The function returns a generator which produces the positions of the contributing double leaves.
        """
        source_contributors_binary = self._source_contributors(source_mask)
        if target_mask:
            target_contributors_binary = self._target_contributors(target_mask)
            contrib = bit_indices(source_contributors_binary & target_contributors_binary)
        else:
            contrib = bit_indices(source_contributors_binary)
        return contrib

    def __repr__(self):
        return f"Dominance index for {self.num_pairs} of pairs of subexpressions for {self.source} and {self.target}."


def bit_indices(bits):
    """
    Iterate the positions of a binary number which are nonzero. Result is a generator and not a list. 
    All binary numbers are read starting at the least-significant bit. 

    Examples:
    list(bit_indices(0b1011)) = [0, 1, 3]
    list(bit_indices(0b1101001) = [0, 3, 5, 6] 
    """
    while bits:
        lowest = bits & -bits
        yield lowest.bit_length()-1
        bits -= lowest


class ExpansionEvaluation:
    """An evaluator-local compiled sum, with a shared support index."""

    def __init__(self, evaluator, expansion):
        self.evaluator = evaluator
        self.expansion = expansion
        self.terms = tuple(expansion.terms)
        self.morphisms = tuple(double_leaf(expansion.system, expansion.source, expansion.target,
                                          term.source_mask, term.target_mask) for term in self.terms)
        self.index = expansion.support_index
        self.values = lru_cache(maxsize=None)(self.values)

    def values(self, prefix):
        values = tuple(self.evaluator.polynomial(term.coefficient, prefix) for term in self.terms)
        active = sum(1 << i for i, value in enumerate(values) if value)
        return values, active

    def entry(self, row, column, prefix):
        values, active = self.values(prefix)
        contributors = self.index.contributors(column, row)
        run_sum = 0
        for i in contributors:
            if i <= active:
                run_sum += values[i]*self.evaluator.entry(self.morphisms[i], row, column, prefix)
        return run_sum

    def column(self, column, prefix):
        values, active = self.values(prefix)
        result = defaultdict(self.evaluator.domain.zero)

        for i in self.index.contributors(column):
            if i <= active:
                for row, value in self.evaluator.column(self.morphisms[i], column, prefix).items():
                    result[row] += values[i]*value
        return {row: value for row, value in result.items() if value}

