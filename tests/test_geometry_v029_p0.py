"""v0.29 P0 fixed geometry, mappings, D4 and graph invariants."""
from itertools import combinations
import unittest

from lotto7x7 import geometry_v029 as geo


def transforms(cells):
    current = tuple(cells)
    for _ in range(4):
        yield tuple(sorted(current))
        yield tuple(sorted((r, 6 - c) for r, c in current))
        current = tuple((c, 6 - r) for r, c in current)


class TestGeometryV029P0(unittest.TestCase):
    def test_region_count_and_boundary(self):
        self.assertEqual(49, len(geo.POINTS))
        self.assertEqual(5, len(geo.REGIONS["disk_r1"]))
        self.assertEqual(13, len(geo.REGIONS["disk_r2"]))
        self.assertEqual(29, len(geo.REGIONS["disk_r3"]))
        self.assertEqual(16, len(geo.REGIONS["annulus_r2_r3"]))
        self.assertEqual(25, len(geo.REGIONS["diamond_r3"]))
        self.assertEqual(24, len(geo.REGIONS["border"]))
        self.assertEqual(13, len(geo.REGIONS["diagonal_union"]))
        self.assertIn((0, 3), geo.REGION_SETS["disk_r3"])
        self.assertNotIn((0, 0), geo.REGION_SETS["disk_r3"])
        self.assertEqual(49, len(set(geo.REGIONS["disk_r3"]) |
                                 (set(geo.POINTS) - set(geo.REGIONS["disk_r3"]))))

    def test_layout_is_a_bijection(self):
        for method in geo.MAPPING_METHODS:
            seq = geo.mapping_positions(method)
            self.assertEqual(49, len(seq))
            self.assertEqual(49, len(set(seq)))
            self.assertEqual(set(seq), set(geo.POINTS))
        self.assertEqual(geo.POINTS, geo.mapping_positions("sequential"))
        self.assertEqual(geo.mapping_positions("random", 649),
                         geo.mapping_positions("random", 649))
        self.assertNotEqual(geo.mapping_positions("random", 649),
                            geo.mapping_positions("random", 650))

    def test_sequential_bitmap_and_example_region(self):
        output = geo.geometry_features([42, 4, 29, 17, 9, 24])
        self.assertEqual([4, 9, 17, 24, 29, 42], output["ticket"])
        self.assertEqual(4, output["region_occupancies"]["disk_r3"])
        self.assertEqual(6, sum(row.count("1") for row in output["bitmap"]))
        self.assertEqual("0001000", output["bitmap"][0])
        self.assertEqual("0" * 7, output["bitmap"][6])
        self.assertEqual(15, sum(output["graph"]["pair_distance_squared_histogram"]))

    def test_graph_topology_on_two_by_three_rectangle(self):
        output = geo.geometry_features([1, 2, 3, 8, 9, 10])
        self.assertEqual(1, output["graph"]["connected_components"])
        self.assertEqual([6], output["graph"]["component_partition"])
        self.assertEqual(7, output["graph"]["cardinal_edges"])
        self.assertEqual(4, output["graph"]["diagonal_edges"])
        self.assertEqual(6, output["graph"]["graph_cycle_rank"])

    def test_d4_invariants_for_many_legal_point_sets(self):
        index = geo.mapping_positions("sequential")
        for numbers in ([4, 9, 17, 24, 29, 42], [1, 2, 3, 8, 9, 10],
                        [1, 7, 17, 22, 39, 49], [4, 13, 19, 28, 34, 46]):
            cells = tuple(index[n - 1] for n in numbers)
            original = geo.geometry_features(numbers)
            base = original["graph"]
            for changed in transforms(cells):
                inverse = {p: i + 1 for i, p in enumerate(index)}
                ticket = sorted(inverse[p] for p in changed)
                item = geo.geometry_features(ticket)
                for key in ("connected_components", "component_partition",
                            "cardinal_edges", "diagonal_edges", "graph_cycle_rank",
                            "pair_distance_squared_histogram"):
                    self.assertEqual(base[key], item["graph"][key], (numbers, key))
                for key in geo.INVARIANT_REGIONS:
                    self.assertEqual(original["region_occupancies"][key],
                                     item["region_occupancies"][key], key)
                self.assertEqual(original["bounding_spans_sorted"], item["bounding_spans_sorted"])
                self.assertEqual(original["center_squared_radius_sum"], item["center_squared_radius_sum"])
                self.assertEqual(sorted(original["region_occupancies"][k]
                                        for k in geo.ORIENTED_REGIONS),
                                 sorted(item["region_occupancies"][k]
                                        for k in geo.ORIENTED_REGIONS))

    def test_validation_failures(self):
        for numbers in ([1, 2, 3, 4, 5, 5], [1, 2, 3, 4, 5, 50],
                        [1, 2, 3, 4, 5], [1, 2, 3, 4, 5, True]):
            with self.assertRaises(ValueError):
                geo.geometry_features(numbers)
        for mapping in ("unknown", "Random", ""):
            with self.assertRaises(ValueError):
                geo.geometry_features([1, 2, 3, 4, 5, 6], mapping=mapping)

    def test_catalog_digest_and_support(self):
        a = geo.catalog()
        b = geo.catalog()
        self.assertEqual(a["definition_sha256"], b["definition_sha256"])
        self.assertEqual(29, a["regions"]["disk_r3"]["lattice_cell_count"])
        self.assertEqual(15, len(list(combinations(range(6), 2))))
        self.assertIn(72, a["pair_distance_squared_support"])
        self.assertNotIn(0, a["pair_distance_squared_support"])
        self.assertEqual(set(geo.REGIONS), set(a["regions"]))


if __name__ == "__main__":
    unittest.main()
