"""Various unit tests for this project.

Run from this directory::

    python3 -m unittest -v tests


"""

import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from sage.all import AA, QQ, ZZ, PolynomialRing, CoxeterGroup

from quadratic import (
    decode_polynomial,
    decode_scalar,
    encode_polynomial,
    encode_scalar,
    field_data,
    field_from_data,
    golden_field,
    homogeneous_parts,
    phi,
)

from coxeter import (
    CoxeterSystem, 
    monomials, 
    polynomial_dimension,
    mask_to_subexpression,
    subexpression_to_mask,
)

from formats import FormatError, check_integer, read_json, sha256, write_json

from morphisms import (
    Localization,
    Morphism, 
    NonRegularPointError, 
    Operation
)

from leaves import (
    DominanceIndex,
    DoubleLeavesBasis,
    ExpansionEvaluation,
    bit_indices,
    double_leaf,
    expand_double_leaf,
    export_leaf,
    gamma,
    light_leaf,
)

from reconstruct import (
    Expansion,
    ExpansionTerm,
    interpolate,
    interpolation_grid,
    reconstruct,
    solve_coefficients_interpolate,
    solve_coefficients_symbolic,
)

from verify import equal, is_zero_interpolate, is_zero_matrix, is_zero_symbolic

class ArithmeticTests(unittest.TestCase):
    """Verify that decoding/encoding of entries in QQ, Q(phi), or AA is correct."""

    @classmethod
    def setUpClass(cls):
        """Build a small, deterministic set of coefficients shared by all tests.

        Zero exercises empty polynomial encodings, fractions exercise rational
        denominators, and phi exercises the quadratic field's basis. Opposite
        square roots in AA check that the codec distinguishes real roots of
        the same minimal polynomial.
        """
        cls.cases = (
            (QQ, (QQ(0), QQ(1), QQ(-7) / 3)),
            (golden_field(), (golden_field()(0), phi(), QQ(2) / 3 - phi())),
            (AA, (AA(0), AA(-7) / 3, AA(2).sqrt(), -AA(2).sqrt())),
        )

    def test_scalar_round_trips(self):
        """Check that a scalar is recovered after an encode-decode round trip."""
        for field, values in self.cases:
            for value in values:
                with self.subTest(field=field, value=value):
                    encoded = json.loads(json.dumps(encode_scalar(value, field)))
                    decoded = decode_scalar(encoded, field)
                    self.assertEqual(decoded, value)
                    self.assertEqual(decoded.parent(), field)

    def test_univariate_polynomial_round_trips(self):
        """Check that a univariate polynomial is recovered after an encode-decode round trip.
        """
        for field, values in self.cases:
            ring = PolynomialRing(field, "x")
            x = ring.gen()
            for coefficient in values:
                for polynomial in (ring(coefficient), coefficient * x**4 - x + 1):
                    with self.subTest(field=field, polynomial=polynomial):
                        self.assert_polynomial_round_trip(polynomial, ring)

    def test_multivariate_polynomial_round_trips(self):
        """Check that a multivariate polynomial is recovered after an encode-decode round trip.
        """
        for field, values in self.cases:
            ring = PolynomialRing(field, names=("x", "y"))
            x, y = ring.gens()
            for coefficient in values:
                for polynomial in (
                    ring(coefficient),
                    coefficient * x**2 * y - y**3 + x + 1,
                ):
                    with self.subTest(field=field, polynomial=polynomial):
                        self.assert_polynomial_round_trip(polynomial, ring)

    def test_golden_field(self):
        """
        Check that the golden field QQ(phi) is implemented correctly. 
        """
        p = phi()
        self.assertEqual(p*p, p+1)
        self.assertEqual(1/p, p-1)
        self.assertEqual((3+2*p).norm(), 11)
        for v in [0, 1, p, (3+2*p)/7]:
            self.assertEqual(decode_scalar(encode_scalar(v, golden_field()), golden_field()), v)

    def test_field_descriptors(self):
        """Round-trip supported fields and reject unknown or malformed descriptors."""
        for field in (QQ, golden_field(), AA):
            with self.subTest(field=field):
                self.assertIs(field_from_data(field_data(field)), field)
        for data in (None, "QQ", {}, {"kind": "unknown"}, {"kind": "QQ", "extra": 1}):
            with self.subTest(data=data):
                with self.assertRaises(ValueError):
                    field_from_data(data)
        with self.assertRaises(ValueError):
            field_data(ZZ)

    def test_scalar_decoder_rejects_invalid_encodings(self):
        """Reject malformed rationals, wrong fields, and invalid algebraic root records."""
        cases = (
            (None, QQ), ({"q": 1}, QQ), ({"q": "1/2/3"}, QQ),
            ({"a": "1", "b": "1"}, QQ),
            ({"minimal_polynomial": ["-1", "0", "1"], "real_root_index": 0}, AA),
            ({"minimal_polynomial": ["-2", "0", "1"], "real_root_index": 2}, AA),
        )
        for data, field in cases:
            with self.subTest(data=data, field=field):
                with self.assertRaises(ValueError):
                    decode_scalar(data, field)

    def test_scalar_decoder_rejects_unknown_record(self):
        """Unknown scalar records must raise an error rather than silently return None."""
        for field in (QQ, golden_field(), AA):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    decode_scalar({"unexpected": "1"}, field)

    def test_polynomial_decoder_rejects_malformed_monomials(self):
        """Reject bad record shapes, negative exponents, and incorrect variable counts."""
        ring = PolynomialRing(QQ, names=("x", "y"))
        cases = ({}, [1], [{"powers": [0, 0]}],
                 [{"powers": [-1, 0], "coefficient": {"q": "1"}}],
                 [{"powers": [1], "coefficient": {"q": "1"}}])
        for data in cases:
            with self.subTest(data=data):
                with self.assertRaises(ValueError):
                    decode_polynomial(data, ring)

    def test_polynomial_decoder_rejects_duplicate_monomials(self):
        """Reject duplicate powers in both univariate and multivariate input."""
        for names in (("x",), ("x", "y")):
            with self.subTest(names=names):
                ring = PolynomialRing(QQ, names=names)
                record = {"powers": [1]*len(names), "coefficient": {"q": "2"}}
                with self.assertRaises(ValueError):
                    decode_polynomial([record, copy.deepcopy(record)], ring)

    def test_polynomial_homogeneous_parts(self):
        """Split univariate and multivariate polynomials exactly, including zero."""
        for names in (("x",), ("x", "y")):
            with self.subTest(names=names):
                ring = PolynomialRing(QQ, names=names)
                x, y = ring.gens()[0], ring.gens()[-1]
                polynomial = 3 + 2*x + x*y
                self.assertEqual(homogeneous_parts(polynomial), {0: ring(3), 1: 2*x, 2: x*y})
                self.assertEqual(homogeneous_parts(ring.zero()), {})

    def assert_polynomial_round_trip(self, polynomial, ring):
        """Check JSON compatibility, exact reconstruction, and the parent ring.

        Also require the zero polynomial to encode as an empty monomial list.
        ``ring`` is the Sage polynomial ring supplied to the decoder.
        """
        encoded = json.loads(json.dumps(encode_polynomial(polynomial)))
        if not polynomial:
            self.assertEqual(encoded, [])
        decoded = decode_polynomial(encoded, ring)
        self.assertEqual(decoded, polynomial)
        self.assertEqual(decoded.parent(), ring)


class FormatTests(unittest.TestCase):
    def test_integer_validation(self):
        """Accept Python and Sage integers while rejecting booleans and coercible nonintegers."""
        for value in (0, -2, ZZ(7)):
            self.assertEqual(check_integer(value, "value"), int(value))
        for value in (True, False, 1.0, "1", None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    check_integer(value, "value")

    def test_json_round_trip_and_replacement(self):
        """Create parent directories, round-trip JSON, and replace an existing file."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "record.json"
            for data in ({"a": [1, 2], "b": {"c": "value"}}, {"replacement": True}):
                self.assertEqual(write_json(path, data), path)
                self.assertEqual(read_json(path), data)
                self.assertFalse(path.with_name(path.name + ".tmp").exists())

    def test_json_rejects_duplicate_keys_and_nonobjects(self):
        """Reject duplicate keys at any depth and JSON whose top level is not an object."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            for text in ('{"a": 1, "a": 2}', '{"a": {"b": 1, "b": 2}}', '[]', 'null'):
                with self.subTest(text=text):
                    path.write_text(text, encoding="utf-8")
                    with self.assertRaises(FormatError):
                        read_json(path)

    def test_digest_is_independent_of_key_order(self):
        """Hash equivalent objects identically while distinguishing changed values."""
        self.assertEqual(sha256({"a": 1, "b": 2}), sha256({"b": 2, "a": 1}))
        self.assertNotEqual(sha256({"a": 1}), sha256({"a": 2}))
        with self.assertRaises(ValueError):
            sha256({"a": float("nan")})


class CoxeterTests(unittest.TestCase):

    def test_mask_conventions_and_round_trips(self):
        """Use least-significant bits first and round-trip all masks through length five."""
        self.assertEqual(subexpression_to_mask((1, 1, 0, 0, 1)), (19, 5))
        for length in range(6):
            for mask in range(1 << length):
                with self.subTest(length=length, mask=mask):
                    self.assertEqual(subexpression_to_mask(mask_to_subexpression(mask, length)), (mask, length))
        for mask, length in ((-1, 2), (4, 2), (0, -1), (True, 2), (0, 1.5)):
            with self.subTest(mask=mask, length=length):
                with self.assertRaises(ValueError):
                    mask_to_subexpression(mask, length)
        with self.assertRaises(ValueError):
            subexpression_to_mask((0, 2))

    def test_monomial_order_and_dimensions(self):
        """Check lexicographic monomial order, rank-one cases, and invalid dimensions."""
        self.assertEqual(monomials(2, 3), ((0, 0, 2), (0, 1, 1), (0, 2, 0),
                                               (1, 0, 1), (1, 1, 0), (2, 0, 0)))
        self.assertEqual(monomials(4, 1), ((4,),))
        self.assertEqual(monomials(0, 3), ((0, 0, 0),))
        self.assertEqual(polynomial_dimension(-1, 3), 0)
        for rank in range(1, 5):
            for degree in range(4):
                self.assertEqual(len(monomials(degree, rank)), polynomial_dimension(degree, rank))
        for function in (monomials, polynomial_dimension):
            for degree, rank in ((1, 0), (True, 2), (1, 1.5)):
                with self.subTest(function=function.__name__, degree=degree, rank=rank):
                    with self.assertRaises(ValueError):
                        function(degree, rank)
        with self.assertRaises(ValueError):
            monomials(-1, 3)

    def test_words_and_canonical_reduced_expressions(self):
        """Normalize identity words and choose reduced words in declared generator order."""
        S = CoxeterSystem("A3")
        self.assertEqual(S.word("e"), ())
        self.assertEqual(S.word(""), ())
        self.assertEqual(S.reduced_word(S.element("tst")), S.word("sts"))
        self.assertEqual(S.reduced_word(S.element("ss")), ())
        with self.assertRaises(ValueError):
            S.word("x")
        custom = CoxeterSystem([[1, 3], [3, 1]], labels=("z", "a"))
        self.assertEqual(custom.reduced_word(custom.element("aza")), tuple("zaz"))
        identity_label = CoxeterSystem([[1]], labels=("e",))
        self.assertEqual(identity_label.word("e"), ("e",))

    def test_deterministic_rex_path(self):
        """Check the prescribed shortest braid path and reject nonreduced or unequal endpoints."""
        S = CoxeterSystem("A3")
        path = S.rex_path("stusts", "utustu")
        self.assertEqual(path, ((2, 2), (0, 3), (2, 3), (1, 2), (4, 2), (2, 3), (0, 3)))
        word = S.word("stusts")
        for position, length in path:
            neighbors = {(p, m): neighbor for neighbor, p, m in S.braid_neighbors(word)}
            word = neighbors[position, length]
        self.assertEqual(word, S.word("utustu"))
        self.assertEqual(S.rex_path("sts", "sts"), ())
        for source, target in (("ss", "ss"), ("s", "t")):
            with self.subTest(source=source, target=target):
                with self.assertRaises(ValueError):
                    S.rex_path(source, target)

    def test_subexpression_decorations_and_defect_counts(self):
        """Check all four decorations and the full defect distribution for a repeated generator."""
        S = CoxeterSystem("A3")
        expected = (("U0", "U0"), ("U1", "D0"), ("U0", "U1"), ("U1", "D1"))
        self.assertEqual(tuple(e.decorations for e in S.subexpressions("ss")), expected)
        data = S.subexpression("sts", 5)
        self.assertEqual(data.bruhat_stroll, (S.element("s"), S.element("s"), S.identity))
        self.assertEqual(data.decorations, ("U1", "U0", "D1"))
        self.assertEqual(data.defect, 1)
        self.assertEqual(S.defect_counts("sss"), {
            (S.identity, 3): 1, (S.identity, 1): 2, (S.identity, -1): 1,
            (S.element("s"), -2): 1, (S.element("s"), 0): 2, (S.element("s"), 2): 1,
        })

    def test_realization_action_and_serialization(self):
        """Preserve A3, B3, and H3 realizations and verify their polynomial group actions."""
        for name in ("A3", "B3", "H3"):
            with self.subTest(system=name):
                S = CoxeterSystem(name)
                data = json.loads(json.dumps(S.to_data()))
                restored = CoxeterSystem.from_data(data)
                self.assertEqual(restored.to_data(), data)
                self.assertEqual(restored.fingerprint, S.fingerprint)
                a, b, _ = S.ring.gens()
                s, t = S.element("s"), S.element("t")
                polynomial = a*b + 2*a + 1
                self.assertEqual(S.act(s, a), -a)
                self.assertEqual(S.act(s, S.act(s, polynomial)), polynomial)
                self.assertEqual(S.act(S.multiply(s, t), polynomial), S.act(s, S.act(t, polynomial)))

    def test_infinite_braid_order(self):
        """Handle an infinite dihedral edge without inventing a finite braid relation."""
        S = CoxeterSystem([[1, None], [None, 1]], labels=("s", "t"))
        self.assertIsNone(S.braid_order("s", "t"))
        self.assertEqual(list(S.braid_neighbors("stst")), [])
        self.assertEqual(S.length(S.element("stst")), 4)
        self.assertEqual(CoxeterSystem.from_data(S.to_data()).to_data(), S.to_data())

    def test_native_sage_group(self):
        """
        Test the basic behavior of our implementation of Coxeter groups from Sage
        In particular, make sure that implemented subexpressions and their defects behave correctly
        """
        W = CoxeterGroup(["A", 4])
        S = CoxeterSystem.from_sage(W)
        self.assertIs(S.group, W)
        self.assertEqual(S.element((1, 2, 1)), W.from_reduced_word([1, 2, 1]))
        self.assertEqual(S.reduced_word(S.element((2, 1, 2))), (1, 2, 1))
        self.assertTrue(S.bruhat_le(S.element((1,)), S.element((2, 1, 2))))
        self.assertEqual(S.subexpression((1, 1), ZZ(0)).defect, 2)
        self.assertEqual(S.subexpression((1, 1), ZZ(1)).defect, -1)
        self.assertEqual(S.subexpression((1, 1), ZZ(3)).defect, 0)

    def test_hom_dimension_nonreduced(self):
        """
        Test whether the graded Hom dimension function produces the correct values.
        """
        S = CoxeterSystem("A3")
        # End(B_s B_s)^0: one degree -2 leaf and three degree 0 leaves.
        p = S.hom_profile("ss", "ss")
        self.assertEqual(p["dimension"], 6)
        self.assertEqual(p["basis_degrees"], {-2: 1, 0: 3})

class MorphismTests(unittest.TestCase):
    """Check exact localization identities and construction of morphisms."""

    def setUp(self):
        """Use A3 and a symbolic evaluator."""
        self.system = CoxeterSystem('A3')
        self.loc = Localization(self.system)
        self.s = 's'
        self.identity = Morphism.identity(self.system, (self.s,))

    def assert_maps_equal(self, first, second):
        """Test whether maps are equal by checking their localization matrices."""
        self.assertEqual((first.source, first.target), (second.source, second.target))
        self.assertEqual(self.loc.localization_matrix(first), self.loc.localization_matrix(second))

    def test_arbitrary_labels(self):
        """Construct all generators."""
        for labels in ((1, 2), ('long_s', 'long_t'), ('e', 't')):
            system = CoxeterSystem([[1, 3], [3, 1]], labels=labels)
            s, t = labels
            expected = {
                'upper_dot': ((s,), ()), 'lower_dot': ((), (s,)),
                'merge': ((s, s), (s,)), 'split': ((s,), (s, s)),
                'cup': ((), (s, s)), 'cap': ((s, s), ()),
            }
            for name, boundaries in expected.items():
                with self.subTest(labels=labels, generator=name):
                    f = getattr(Morphism, name)(system, s)
                    self.assertEqual((f.source, f.target), boundaries)
                    self.assertEqual(f.dual().dual().to_data(), f.to_data())
            f = Morphism.braid(system, s, t)
            self.assertEqual((f.source, f.target), ((s, t, s), (t, s, t)))

    def test_one_color_relations(self):
        """Verify units, counits, needle, circle, and both zig-zag identities."""
        S, s, I = self.system, self.s, self.identity
        up, down = Morphism.upper_dot(S, s), Morphism.lower_dot(S, s)
        merge, split = Morphism.merge(S, s), Morphism.split(S, s)
        cup, cap = Morphism.cup(S, s), Morphism.cap(S, s)
        self.assert_maps_equal(down.then(up), Morphism.box(S, (), S.ring.gen(0)))
        self.assert_maps_equal(down.then(split), cup)
        self.assert_maps_equal(merge.then(up), cap)
        for f in (down.tensor(I), I.tensor(down)):
            self.assert_maps_equal(f.then(merge), I)
        for f in (up.tensor(I), I.tensor(up)):
            self.assert_maps_equal(split.then(f), I)
        self.assert_maps_equal(split.then(merge), Morphism.zero(S, (s,), (s,)))
        self.assert_maps_equal(cup.then(cap), Morphism.zero(S, (), ()))
        self.assert_maps_equal(cup.tensor(I).then(I.tensor(cap)), I)
        self.assert_maps_equal(I.tensor(cup).then(cap.tensor(I)), I)

    def test_frobenius_relations(self):
        """Verify associativity, coassociativity, and the Frobenius associativity."""
        S, s, I = self.system, self.s, self.identity
        merge, split = Morphism.merge(S, s), Morphism.split(S, s)
        self.assert_maps_equal(merge.tensor(I).then(merge), I.tensor(merge).then(merge))
        self.assert_maps_equal(split.then(split.tensor(I)), split.then(I.tensor(split)))
        self.assert_maps_equal(merge.then(split), split.tensor(I).then(I.tensor(merge)))
        self.assert_maps_equal(merge.then(split), I.tensor(split).then(merge.tensor(I)))

    def test_polynomial_regions_and_prefix(self):
        """Test action of W on polynomial boxes."""
        S, I = self.system, self.identity
        p = S.ring.gen(1)
        right = Morphism.box(S, I.source, p, region=1)
        self.assert_maps_equal(right, I.tensor(Morphism.box(S, (), p)))
        self.assert_maps_equal(I.scale(p), Morphism.box(S, I.source, p))
        self.assertEqual(self.loc.entry(right, 1, 1), S.act(S.simple['s'], p))
        prefix = S.simple['t']
        for row in (0, 1):
            self.assertEqual(self.loc.entry(right, row, row, prefix),
                             S.act(prefix, self.loc.entry(right, row, row)))

    def test_positioned_generators(self):
        """Test that an operation is a generator placed at the correct position padded by identity strands."""
        S = self.system
        left, right = Morphism.identity(S, 't'), Morphism.identity(S, 'u')
        for name in ('upper_dot', 'lower_dot', 'merge', 'split', 'cup', 'cap'):
            f = getattr(Morphism, name)(S, 's')
            operation = f.payload[0]
            padded = Morphism.generator(S, left.source + f.source + right.source,
                                        operation.kind, 1, color=operation.color)
            self.assert_maps_equal(padded, left.tensor(f).tensor(right))

    def test_braid_coefficients(self):
        """Verify that the localization matrix for the 6-valent vertex is correct."""
        S = self.system
        commuting = Morphism.braid(S, 's', 'u')
        self.assertEqual(dict(self.loc.localization_matrix(commuting)),
                         {(0, 0): 1, (2, 1): 1, (1, 2): 1, (3, 3): 1})
        braid = Morphism.braid(S, 's', 't')
        a, b = S.ring.gens()[:2]
        self.assertEqual(self.loc.entry(braid, 0, 0), (a+b)/b)
        for system in (S, CoxeterSystem('H3')):
            loc = Localization(system)
            f = Morphism.braid(system, 's', 't')
            top = (1 << len(f.source))-1
            self.assertEqual(loc.entry(f, top, top), 1)
            for (r, c), value in loc.localization_matrix(f).items():
                self.assertTrue(value)
                self.assertEqual(system.subexpression(f.source, c).endpoint,
                                 system.subexpression(f.target, r).endpoint)

    def test_duality_and_composition(self):
        """Check duals on graphs and compositions."""
        S, I = self.system, self.identity
        p = S.ring.gen(0)
        f = (I + I.scale(p)).then(Morphism.split(S, 's'))
        self.assertEqual(f.kind, 'compose')
        self.assert_maps_equal(f.dual().dual(), f)
        self.assert_maps_equal(f @ I, I.then(f))
        self.assert_maps_equal(f.dual(), Morphism.merge(S, 's').then((I+I.scale(p)).dual()))
        graph = Morphism.cup(S, 's').append('upper dot', 1)
        self.assert_maps_equal(graph.dual().dual(), graph)
        braid = Morphism.braid(S, 's', 't')
        self.assert_maps_equal(braid.dual(), Morphism.braid(S, 't', 's'))

    def test_homogeneous_decomposition(self):
        """Check degree and homogeneous decomposition."""
        S, I = self.system, self.identity
        x = S.ring.gen(0)
        f = (I.scale(1+x)).then(Morphism.split(S, 's')).tensor(I.scale(x))
        self.assertEqual(set(f.homogeneous_components), {1, 3})
        self.assert_maps_equal(sum(f.homogeneous_components.values()), f)
        self.assertIsNone(f.degree)
        self.assertEqual((I-I).degree, 0)
        self.assertEqual(Morphism.zero(S, 's', 's').homogeneous_components, {})
        box = Morphism.elias_williamson_graph(S, (), [Operation('polynomial', 0, polynomial=2)])
        self.assertEqual(box.degree, 0)
        self.assertEqual(Morphism.from_data(box.to_data(), system=S).degree, 0)

    def test_serialization(self):
        """Encode and then decode a morphism through JSON."""
        S, I = self.system, self.identity
        f = (I+I).tensor(Morphism.braid(S, 's', 't')).scale(S.ring.gen(0)+1)
        data = json.loads(json.dumps(f.to_data()))
        recovered = Morphism.from_data(data, system=S)
        self.assert_maps_equal(recovered, f)
        shared_sum = recovered.children[0].children[0]
        self.assertIs(shared_sum.children[0], shared_sum.children[1])
        with tempfile.TemporaryDirectory() as directory:
            path = f.save(Path(directory)/'morphism.json')
            self.assert_maps_equal(Morphism.load(path, system=S), f)

    def test_exact_specialization(self):
        """Compare interpolation point evaluation to specializing symbolic coefficients."""
        S = self.system
        point = (2, 3, 5)
        numeric = Localization(S, point=point)
        f = Morphism.braid(S, 's', 't').tensor(Morphism.cup(S, 'u'))
        symbolic = self.loc.localization_matrix(f)
        expected = {key: value(*point) for key, value in symbolic.items() if value(*point)}
        self.assertEqual(numeric.localization_matrix(f), expected)
        with self.assertRaises(NonRegularPointError):
            Localization(S, point=(0, 1, 1)).column(Morphism.cup(S, 's'), 0)

    def test_lambda_value_wrapper(self):
        """Check symbolic and specialized lambda values and explicit evaluator precedence."""
        S = self.system
        a = S.ring.gen(0)
        morphism = self.identity.scale(a)
        basis = DoubleLeavesBasis(S, "s", "s", degree=2)
        index = next(i for i, pair in enumerate(basis.entries) if pair.key == (1, 1))
        self.assertEqual(morphism.lambda_value(basis, index), a)
        self.assertEqual(morphism.lambda_value(basis, index, point=(2, 3, 1)), 2)
        self.assertEqual(morphism.lambda_value(basis, index, point=(2, 3, 1), evaluator=self.loc), a)

    def test_tensor_interchange(self):
        """Composition on independent tensor factors agrees with composing each factor first."""
        S = self.system
        first, second = Morphism.split(S, "s"), Morphism.lower_dot(S, "t")
        after_first, after_second = Morphism.cap(S, "s"), Morphism.split(S, "t")
        self.assert_maps_equal(
            first.tensor(second).then(after_first.tensor(after_second)),
            first.then(after_first).tensor(second.then(after_second)),
        )

    def test_cache_and_readonly_results(self):
        """Bound each evaluator cache and prevent mutations of cached results."""
        loc = Localization(self.system, cache_size=ZZ(2))
        for name in ('root', 'zeta', 'polynomial', '_braid', '_column', '_double_leaf', '_expansion_index'):
            self.assertEqual(getattr(loc, name).cache_info().maxsize, 2)
        f = Morphism.identity(self.system, 'ss')
        for column in range(4):
            loc.column(f, column)
        self.assertEqual(loc._column.cache_info().currsize, 2)
        with self.assertRaises(TypeError):
            loc.column(f, 0)[0] = 7
        with self.assertRaises(ValueError):
            Localization(self.system, cache_size=True)

    def test_invalid_inputs(self):
        """Reject malformed positions, records, boundaries, and masks in the construction of morphisms."""
        S = self.system
        with self.assertRaises(ValueError):
            Morphism.generator(S, 'sts', 'braid', -1)
        with self.assertRaises(ValueError):
            Operation.from_data({'kind': 'cup'}, S)
        with self.assertRaises(ValueError):
            self.loc.column(self.identity, 2)
        with self.assertRaises(ValueError):
            self.identity.then(Morphism.identity(S, 't'))
        with self.assertRaises(ValueError):
            Morphism.generator(S, 'st', 'merge')
        with self.assertRaises(ValueError):
            Morphism.braid(S, 's', 's')

class AlgebraTestCase(unittest.TestCase):
    def setUp(self):
        self.system = CoxeterSystem("A3")
        self.loc = Localization(self.system)
        self.alpha, self.beta, self.delta = self.system.ring.gens()

    def assert_maps_equal(self, left, right):
        """Use full matrices as an oracle independent of triangular solving."""
        self.assertEqual((left.source, left.target), (right.source, right.target))
        self.assertEqual(
            self.loc.localization_matrix(left), self.loc.localization_matrix(right)
        )


class LeavesTests(AlgebraTestCase):
    def test_light_leaf_elementary_steps(self):
        """Check empty, dot, identity, merge, and cap light leaves and their degrees."""
        S = self.system
        cases = (
            ("", 0, Morphism.identity(S, "")),
            ("s", 0, Morphism.upper_dot(S, "s")),
            ("s", 1, Morphism.identity(S, "s")),
            ("ss", 1, Morphism.merge(S, "s")),
            ("ss", 3, Morphism.cap(S, "s")),
        )
        for word, mask, expected in cases:
            with self.subTest(word=word, mask=mask):
                leaf = light_leaf(S, word, mask)
                self.assert_maps_equal(leaf, expected)
                self.assertEqual(leaf.degree, S.subexpression(word, mask).defect)

    def test_light_leaf_canonical_target_and_braid(self):
        """Check canonical reduced targets and degrees, including a nontrivial braid."""
        S = self.system
        self.assert_maps_equal(light_leaf(S, "tst", 7), Morphism.braid(S, "t", "s"))
        for mask in range(16):
            with self.subTest(mask=mask):
                data = S.subexpression("stst", mask)
                leaf = light_leaf(S, "stst", mask)
                self.assertEqual(leaf.source, S.word("stst"))
                self.assertEqual(leaf.target, S.reduced_word(data.endpoint))
                self.assertEqual(leaf.degree, data.defect)

    def test_double_leaf_weighted_entries_match_explicit_graphs(self):
        """Compare weighted double-leaf matrices with their expanded elementary graphs."""
        S = self.system
        # Nonreduced words include both merge and cap light leaves.
        for e in S.subexpressions("sts"):
            for f in S.subexpressions("ss"):
                if e.endpoint != f.endpoint:
                    continue
                with self.subTest(source_mask=e.mask, target_mask=f.mask):
                    leaf = double_leaf(S, "sts", "ss", e.mask, f.mask)
                    self.assertEqual(leaf.payload, (e.mask, f.mask))
                    self.assertEqual(leaf.degree, e.defect + f.defect)
                    self.assert_maps_equal(leaf, expand_double_leaf(leaf))

    def test_invalid_leaf_inputs(self):
        """Reject out-of-range masks, unequal endpoints, and expansion of a non-leaf."""
        S = self.system
        with self.assertRaises(ValueError):
            light_leaf(S, "s", 2)
        with self.assertRaises(ValueError):
            double_leaf(S, "s", "s", 0, 1)
        with self.assertRaises(ValueError):
            expand_double_leaf(Morphism.identity(S, "s"))

    def test_leaf_export_round_trip(self):
        """Save light and double leaves as graphs and recover their localization matrices."""
        S = self.system
        leaves = (light_leaf(S, "ss", 3), double_leaf(S, "ss", "s", 1, 1))
        with tempfile.TemporaryDirectory() as directory:
            for i, leaf in enumerate(leaves):
                with self.subTest(kind=leaf.kind):
                    path = Path(directory) / f"leaf-{i}.json"
                    export_leaf(leaf, path)
                    loaded = Morphism.load(path, system=S)
                    self.assertEqual(loaded.kind, "Elias-Williamson graph")
                    self.assert_maps_equal(loaded, leaf)

    def test_basis_degrees_and_hom_dimension(self):
        """Check the nonreduced basis size, coefficient degrees, and graded Hom dimension."""
        basis = DoubleLeavesBasis(self.system, "ss", "ss")
        self.assertEqual(len(basis), 4)
        self.assertEqual(sorted(p.degree for p in basis.entries), [-2, 0, 0, 0])
        self.assertEqual(sorted(basis.coefficient_degrees), [0, 0, 0, 1])
        self.assertEqual(
            sum(polynomial_dimension(d, self.system.rank) for d in basis.coefficient_degrees),
            6,
        )
        self.assertEqual(len(DoubleLeavesBasis(self.system, "", "", degree=-1)), 0)
        with self.assertRaises(ValueError):
            DoubleLeavesBasis(self.system, "s", "s", degree=True)

    def test_lambda_matrix_is_lower_triangular_with_prescribed_diagonal(self):
        """Check triangularity and compare each nonzero pivot with the diagonal formula."""
        basis = DoubleLeavesBasis(self.system, "ss", "ss")
        for j in range(len(basis)):
            values = tuple(basis.lambda_values(basis.morphism(j), self.loc))
            for i, value in enumerate(values):
                with self.subTest(row=i, column=j):
                    if j > i:
                        self.assertEqual(value, 0)
                    elif i == j:
                        self.assertNotEqual(value, 0)
                        self.assertEqual(value, basis.diagonal(i, self.loc))
            self.assertEqual(basis[j].key, basis.morphism(j).payload)

    def test_lambda_rejects_incompatible_boundaries_and_systems(self):
        """Reject lambda evaluation with mismatched boundaries or realizations."""
        basis = DoubleLeavesBasis(self.system, "s", "s")
        identity = Morphism.identity(self.system, "s")
        with self.assertRaises(ValueError):
            basis.lambda_value(Morphism.identity(self.system, "t"), 0, self.loc)
        with self.assertRaises(ValueError):
            basis.lambda_value(identity, 0, Localization(CoxeterSystem("A3")))

    def test_gamma_signs_and_prefix_action(self):
        """Check gamma for empty words, down steps, and a reflecting prefix."""
        S = self.system
        self.assertEqual(gamma(self.loc, "", 0), 1)
        self.assertEqual(gamma(self.loc, "s", 0), self.alpha)
        self.assertEqual(gamma(self.loc, "ss", 1), 1)
        self.assertEqual(gamma(self.loc, "ss", 3), self.alpha)
        self.assertEqual(gamma(self.loc, "s", 0, S.element("s")), -self.alpha)

    def test_dominance_index_filters_both_paths(self):
        """Check source-only and joint path filtering, including zero masks and empty support."""
        S = self.system
        pairs = ((0, 0), (1, 0), (0, 1), (1, 1))
        index = DominanceIndex(S, S.word("s"), S.word("s"), pairs)
        self.assertEqual(list(index.contributors(0)), [0, 1, 2, 3])
        self.assertEqual(list(index.contributors(1)), [1, 3])
        self.assertEqual(list(index.contributors(0, 1)), [2, 3])
        self.assertEqual(list(index.contributors(1, 1)), [3])
        self.assertEqual(list(index.contributors(1, 0)), [1, 3])
        self.assertEqual(list(DominanceIndex(S, (), (), ()).contributors(0, 0)), [])

    def test_bit_indices(self):
        """Enumerate set bits in increasing order, including empty and large bitsets."""
        for bits, expected in ((0, []), (1, [0]), (0b1011, [0, 1, 3]), (1 << 80, [80])):
            with self.subTest(bits=bits):
                self.assertEqual(list(bit_indices(bits)), expected)

    def test_expansion_evaluation_matches_explicit_sum(self):
        """Compare compiled entries and columns with explicit sums at symbolic and exact points."""
        S = self.system
        expansion = Expansion(S, "ss", "s", (
            ExpansionTerm(1, 1, self.alpha - 2*self.delta), ExpansionTerm(2, 1, 2),
            ExpansionTerm(3, 0, self.beta - 3*self.delta),
        ))
        # At (2, 3, 1), only the middle coefficient survives specialization.
        explicit = sum(double_leaf(S, "ss", "s", *t.key).scale(t.coefficient)
                       for t in expansion.terms)
        for loc in (self.loc, Localization(S, point=(2, 3, 1))):
            compiled = ExpansionEvaluation(loc, expansion)
            for prefix in (S.identity, S.element("s")):
                for column in range(4):
                    with self.subTest(point=loc.point, prefix=prefix, column=column):
                        expected = loc.column(explicit, column, prefix)
                        self.assertEqual(compiled.column(column, prefix), expected)
                        for row in range(2):
                            self.assertEqual(compiled.entry(row, column, prefix), expected.get(row, 0))


class ReconstructionTests(AlgebraTestCase):
    def test_rank_one_reconstruction(self):
        """Recover a higher-degree coefficient from the single rank-one interpolation point."""
        S = CoxeterSystem([[1]], labels=("s",))
        polynomial = 3*S.ring.gen()**4
        morphism = Morphism.identity(S, "s").scale(polynomial)
        for method in ("symbolic", "interpolate"):
            with self.subTest(method=method):
                expansion = reconstruct(morphism, method=method)
                self.assertEqual(expansion.terms, (ExpansionTerm(1, 1, polynomial),))
                loc = Localization(S)
                self.assertEqual(loc.localization_matrix(expansion.as_morphism()), loc.localization_matrix(morphism))

    def test_golden_field_reconstruction_and_integrality(self):
        """Recover exact Q(phi) coefficients and distinguish integral from fractional expansions."""
        S = CoxeterSystem("H3")
        polynomial = phi()*S.ring.gen(0) + (phi() - 1)*S.ring.gen(1)
        morphism = Morphism.identity(S, "s").scale(polynomial)
        for method in ("symbolic", "interpolate"):
            with self.subTest(method=method):
                expansion = reconstruct(morphism, method=method)
                self.assertEqual(expansion.terms, (ExpansionTerm(1, 1, polynomial),))
                self.assertTrue(expansion.is_integral())
                restored = Expansion.from_data(json.loads(json.dumps(expansion.to_data())), system=S)
                self.assertEqual(restored.terms, expansion.terms)
        fractional = Expansion(S, "s", "s", (ExpansionTerm(1, 1, polynomial/2),))
        self.assertFalse(fractional.is_integral())

    def test_interpolation_grid(self):
        """Check grid ordering, the rank-one special case, and sample counts by degree."""
        self.assertEqual(interpolation_grid(1, 5), ((0, (1,)),))
        self.assertEqual(interpolation_grid(3, 1), (
            (0, (2, 3, 1)), (1, (2, 4, 1)), (1, (3, 3, 1)),
        ))
        for degree in range(4):
            self.assertEqual(len(interpolation_grid(3, degree)), polynomial_dimension(degree, 3))

    def test_interpolate_exact_homogeneous_polynomials(self):
        """Recover exact constant, linear, and quadratic polynomials and reject too few samples."""
        polynomials = (self.system.ring(7), self.alpha - 2*self.beta,
                       QQ(2)/3*self.alpha**2 + self.beta*self.delta - 4*self.delta**2)
        for degree, polynomial in enumerate(polynomials):
            with self.subTest(degree=degree):
                points = [p for _, p in interpolation_grid(3, degree)]
                values = [polynomial(*p) for p in points]
                self.assertEqual(interpolate(self.system, degree, points, values), polynomial)
        with self.assertRaises(ValueError):
            interpolate(self.system, 1, [(2, 3, 1)], [1])

    def test_reconstruct_known_three_term_relation(self):
        """Recover the known three-term coefficients with both solvers and check progress reports."""
        S = self.system
        morphism = Morphism.identity(S, "s").tensor(Morphism.upper_dot(S, "s"))
        expected = {(1, 1): -self.alpha, (2, 1): 1, (3, 0): 1}
        for method in ("symbolic", "interpolate"):
            with self.subTest(method=method):
                reports = []
                expansion = reconstruct(morphism, method=method, progress=reports.append, progress_every=1)
                self.assertEqual({t.key: t.coefficient for t in expansion.terms}, expected)
                self.assert_maps_equal(morphism, expansion.as_morphism())
                self.assertEqual(expansion.metadata["algorithm"], method)
                self.assertTrue(any(isinstance(report, dict) for report in reports))

    def test_solvers_return_coefficients_in_basis_order(self):
        """Recover prescribed coefficients, including zero, in the declared basis order."""
        basis = DoubleLeavesBasis(self.system, "ss", "s", degree=1)
        expected = {(1, 1): self.alpha, (2, 1): 0, (3, 0): -2}
        morphism = sum(basis.morphism(i).scale(expected[p.key]) for i, p in enumerate(basis.entries))
        for solve in (solve_coefficients_symbolic, solve_coefficients_interpolate):
            with self.subTest(solver=solve.__name__):
                self.assertEqual(solve(morphism, basis), tuple(expected[p.key] for p in basis.entries))

    def test_reconstruct_zero_and_inhomogeneous_morphisms(self):
        """Reconstruct zero and mixed-degree maps and check the default algorithm."""
        S = self.system
        identity = Morphism.identity(S, "s")
        polynomial = 1 + self.alpha + self.beta**2
        for method in ("symbolic", "interpolate"):
            with self.subTest(method=method):
                zero = reconstruct(Morphism.zero(S, "s", "s"), method=method)
                self.assertEqual(zero.terms, ())
                self.assertEqual(zero.metadata["profile"], [])
                expansion = reconstruct(identity.scale(polynomial), method=method)
                self.assertEqual({t.key: t.coefficient for t in expansion.terms}, {(1, 1): polynomial})
                self.assert_maps_equal(expansion.as_morphism(), identity.scale(polynomial))
        default = reconstruct(identity)
        self.assertEqual(default.metadata["algorithm"], "interpolate")

    def test_reconstruction_rejects_invalid_method(self):
        """Reject an unsupported reconstruction algorithm."""
        with self.assertRaises(ValueError):
            reconstruct(Morphism.identity(self.system, "s"), method="unknown")

    def test_solvers_reject_invalid_inputs(self):
        """Reject incompatible systems, boundaries, degrees, and invalid progress intervals."""
        S = self.system
        basis = DoubleLeavesBasis(S, "s", "s")
        identity = Morphism.identity(S, "s")
        for solve in (solve_coefficients_symbolic, solve_coefficients_interpolate):
            for bad in (Morphism.identity(CoxeterSystem("A3"), "s"),
                        Morphism.identity(S, "t"), identity.scale(self.alpha)):
                with self.subTest(solver=solve.__name__, morphism=bad):
                    with self.assertRaises(ValueError):
                        solve(bad, basis)
            for interval in (0, -1, True, 1.5):
                with self.subTest(solver=solve.__name__, interval=interval):
                    with self.assertRaises(ValueError):
                        solve(identity, basis, progress_every=interval)


class ExpansionTests(AlgebraTestCase):
    def test_combines_duplicates_and_omits_zero_terms(self):
        """Combine repeated leaf coefficients, remove cancellations, and reject invalid masks."""
        expansion = Expansion(self.system, "s", "s", (
            ExpansionTerm(1, 1, self.alpha), ExpansionTerm(1, 1, 2),
            ExpansionTerm(0, 0, 3), ExpansionTerm(0, 0, -3),
        ))
        self.assertEqual(expansion.terms, (ExpansionTerm(1, 1, self.alpha + 2),))
        self.assertIs(expansion.support_index, expansion.support_index)
        with self.assertRaises(ValueError):
            ExpansionTerm(True, 0, 1)
        with self.assertRaises(ValueError):
            Expansion(self.system, "s", "s", (ExpansionTerm(0, 1, 1),))

    def test_homogeneous_components_and_dual(self):
        """Split polynomial coefficients by degree and check reflection and double duality."""
        S = self.system
        expansion = Expansion(S, "ss", "s", (ExpansionTerm(2, 1, 1 + self.alpha),))
        pieces = expansion.homogeneous_components()
        self.assertEqual(set(pieces), {1, 3})
        self.assert_maps_equal(sum(pieces.values()), expansion.as_morphism())
        dual = expansion.dual()
        self.assertEqual((dual.source, dual.target), (expansion.target, expansion.source))
        self.assertEqual(dual.terms, (ExpansionTerm(1, 2, 1 + self.alpha),))
        graph = expand_double_leaf(double_leaf(S, "ss", "s", 2, 1)).scale(1 + self.alpha)
        self.assert_maps_equal(dual.as_morphism(), graph.dual())
        self.assertEqual(dual.dual().terms, expansion.terms)

    def test_json_and_file_round_trips(self):
        """Preserve coefficients, metadata, and realizations through JSON and file round trips."""
        expansion = Expansion(self.system, "s", "s", (
            ExpansionTerm(1, 1, QQ(2)/3*self.alpha + 1),
        ), metadata={"note": "unit test"})
        data = json.loads(json.dumps(expansion.to_data()))
        restored = Expansion.from_data(data, system=self.system)
        self.assertEqual(restored.terms, expansion.terms)
        self.assertEqual(restored.metadata, expansion.metadata)
        self.assertEqual(Expansion.from_data(data).to_data(), data)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "expansion.json"
            expansion.save(path)
            loaded = Expansion.load(path, system=self.system)
            self.assertEqual(loaded.to_data(), expansion.to_data())

    def test_rejects_corrupt_serialized_terms(self):
        """Reject incorrect leaf metadata, duplicate or zero terms, and realization mismatches."""
        expansion = Expansion(self.system, "s", "s", (ExpansionTerm(1, 1, 1),))
        for key, value in (("middle_word", []), ("leaf_degree", 2),
                           ("coefficient", []), ("source_mask", 0)):
            with self.subTest(key=key):
                data = copy.deepcopy(expansion.to_data())
                data["terms"][0][key] = value
                with self.assertRaises(ValueError):
                    Expansion.from_data(data, system=self.system)
        data = expansion.to_data()
        data["terms"].append(copy.deepcopy(data["terms"][0]))
        with self.assertRaises(ValueError):
            Expansion.from_data(data, system=self.system)
        with self.assertRaises(ValueError):
            Expansion.from_body({}, self.system, "s", "s")
        with self.assertRaises(ValueError):
            Expansion.from_data(expansion.to_data(), system=CoxeterSystem("B3"))

    def test_integrality(self):
        """Distinguish integral coefficients from fractions, including the empty expansion."""
        for coefficient, expected in ((0, True), (2*self.alpha + 3, True), (self.alpha/2, False)):
            with self.subTest(coefficient=coefficient):
                expansion = Expansion(self.system, "s", "s", (ExpansionTerm(1, 1, coefficient),))
                self.assertEqual(expansion.is_integral(), expected)


class VerificationTests(AlgebraTestCase):
    def test_zero_checkers_with_empty_basis(self):
        """Zero maps remain zero in degrees with no contributing double leaves."""
        basis = DoubleLeavesBasis(self.system, "", "", degree=-1)
        zero = Morphism.zero(self.system, "", "")
        self.assertEqual(len(basis), 0)
        for check in (is_zero_symbolic, is_zero_interpolate):
            with self.subTest(checker=check.__name__):
                self.assertTrue(check(zero, basis))

    def test_verification_of_zero_and_empty_word_maps(self):
        """Distinguish scalar maps on the empty word and handle explicit zero maps."""
        S = self.system
        identity = Morphism.identity(S, "")
        zero = Morphism.zero(S, "", "")
        for algorithm in ("matrix", "symbolic", "interpolate"):
            with self.subTest(algorithm=algorithm):
                self.assertTrue(equal(zero, zero, algorithm=algorithm))
                self.assertTrue(equal(identity.scale(self.alpha), Morphism.box(S, "", self.alpha), algorithm=algorithm))
                self.assertFalse(equal(identity, zero, algorithm=algorithm))

    def test_equal_and_unequal_maps_for_all_algorithms(self):
        """Verify a unit relation and detect unequal maps with all three algorithms."""
        S = self.system
        identity = Morphism.identity(S, "s")
        # The Frobenius unit relation is represented by different graphs.
        unit_relation = Morphism.lower_dot(S, "s").tensor(identity).then(Morphism.merge(S, "s"))
        for algorithm in ("matrix", "symbolic", "interpolate"):
            with self.subTest(algorithm=algorithm):
                reports = []
                self.assertTrue(equal(unit_relation, identity, algorithm=algorithm, progress=reports.append))
                self.assertFalse(equal(identity, identity.scale(2), algorithm=algorithm, progress=reports.append))
                self.assertTrue(reports)

    def test_equal_without_progress_callback(self):
        """Verify equal maps without requiring an optional progress callback."""
        identity = Morphism.identity(self.system, "s")
        for algorithm in ("matrix", "symbolic", "interpolate"):
            with self.subTest(algorithm=algorithm):
                self.assertTrue(equal(identity, identity, algorithm=algorithm))

    def test_zero_checkers_detect_vanishing_and_nonvanishing(self):
        """Distinguish formal cancellation and explicit zero from a nonzero identity."""
        identity = Morphism.identity(self.system, "s")
        basis = DoubleLeavesBasis(self.system, "s", "s")
        for check in (is_zero_symbolic, is_zero_interpolate):
            with self.subTest(checker=check.__name__):
                self.assertTrue(check(identity - identity, basis))
                self.assertTrue(check(Morphism.zero(self.system, "s", "s"), basis))
                self.assertFalse(check(identity, basis))
        self.assertTrue(is_zero_matrix(identity - identity))
        self.assertFalse(is_zero_matrix(identity))

    def test_interpolation_checks_more_than_the_first_point(self):
        # This nonzero homogeneous polynomial vanishes at the first grid point.
        morphism = Morphism.identity(self.system, "s").scale(self.alpha - 2*self.delta)
        basis = DoubleLeavesBasis(self.system, "s", "s", degree=2)
        self.assertFalse(is_zero_interpolate(morphism, basis))

    def test_leaf_evaluation_methods_agree(self):
        """Verify the same leaf identity using both evaluation methods and every algorithm."""
        leaf = double_leaf(self.system, "ss", "s", 1, 1)
        for algorithm in ("matrix", "symbolic", "interpolate"):
            for method in ("weighted", "Elias-Williamson graphs"):
                with self.subTest(algorithm=algorithm, leaf_method=method):
                    self.assertTrue(equal(leaf, expand_double_leaf(leaf), algorithm=algorithm,
                                          leaf_method=method, progress=lambda report: None))

    def test_equal_rejects_different_systems_and_boundaries(self):
        """Return false when morphisms have different systems, sources, or targets."""
        S = self.system
        identity = Morphism.identity(S, "s")
        for other in (Morphism.identity(CoxeterSystem("A3"), "s"),
                      Morphism.zero(S, "t", "s"), Morphism.zero(S, "s", "t")):
            with self.subTest(other=other):
                self.assertFalse(equal(identity, other))

    def test_invalid_verification_options(self):
        """Reject unsupported verification algorithms and leaf evaluation methods."""
        identity = Morphism.identity(self.system, "s")
        for options in ({"algorithm": "unknown"}, {"leaf_method": "unknown"}):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    equal(identity, identity, **options)

    def test_zero_checkers_reject_incompatible_inputs(self):
        """Reject incompatible zero-checking inputs and invalid progress intervals."""
        S = self.system
        basis = DoubleLeavesBasis(S, "s", "s")
        identity = Morphism.identity(S, "s")
        for check in (is_zero_symbolic, is_zero_interpolate):
            for bad in (Morphism.identity(CoxeterSystem("A3"), "s"),
                        Morphism.identity(S, "t"), identity.scale(self.alpha)):
                with self.subTest(checker=check.__name__, morphism=bad):
                    with self.assertRaises(ValueError):
                        check(bad, basis)
            for interval in (0, -1, True, 1.5):
                with self.subTest(checker=check.__name__, interval=interval):
                    with self.assertRaises(ValueError):
                        check(identity, basis, progress_every=interval)



if __name__ == "__main__":
    unittest.main()
