import unittest

from matplotlib import pyplot
from networkx import Graph

from agh_graphs.productions.p4 import P4
from agh_graphs.utils import gen_name, add_interior, get_neighbors_at
from agh_graphs.visualize import visualize_graph_3d
from tests.utils import visualize_tests


class P4Test(unittest.TestCase):
    def test_happy_path(self):
        graph = Graph()
        e1, e2, e3, e4, e5 = [gen_name() for _ in range(5)]

        graph.add_node(e1, layer=1, position=(1.0, 2.0), label='E')
        graph.add_node(e2, layer=1, position=(1.0, 1.5), label='E')
        graph.add_node(e3, layer=1, position=(1.0, 1.0), label='E')
        graph.add_node(e4, layer=1, position=(2.0, 1.0), label='E')
        graph.add_node(e5, layer=1, position=(1.5, 1.5), label='E')

        graph.add_edge(e1, e2)
        graph.add_edge(e2, e3)
        graph.add_edge(e3, e4)
        graph.add_edge(e4, e5)
        graph.add_edge(e5, e1)

        i = add_interior(graph, e1, e3, e4)

        if visualize_tests:
            visualize_graph_3d(graph)
            pyplot.show()

        [i1, i2, i3] = P4().apply(graph, [i])

        self.assertEqual(len(graph.nodes()), 14)
        self.assertEqual(len(graph.edges()), 27)

        self.assertEqual(graph.nodes[i]['label'], 'i')
        self.assertTrue(graph.has_edge(i, i1))
        self.assertTrue(graph.has_edge(i, i2))
        self.assertTrue(graph.has_edge(i, i3))

        self.assertEqual(graph.nodes[i1]['label'], 'I')
        self.assertEqual(graph.nodes[i2]['label'], 'I')
        self.assertEqual(graph.nodes[i3]['label'], 'I')
        self.assertEqual(graph.nodes[i1]['layer'], graph.nodes[i]['layer'] + 1)
        self.assertEqual(graph.nodes[i2]['layer'], graph.nodes[i]['layer'] + 1)
        self.assertEqual(graph.nodes[i3]['layer'], graph.nodes[i]['layer'] + 1)

        i1_neighbors = get_neighbors_at(graph, i1, graph.nodes[i1]['layer'])
        self.assertEqual(len(i1_neighbors), 3)
        i2_neighbors = get_neighbors_at(graph, i2, graph.nodes[i2]['layer'])
        self.assertEqual(len(i2_neighbors), 3)
        i3_neighbors = get_neighbors_at(graph, i3, graph.nodes[i3]['layer'])
        self.assertEqual(len(i3_neighbors), 3)

        i1_i2_n = [x for x in i1_neighbors if x in i2_neighbors]
        i1_i3_n = [x for x in i1_neighbors if x in i3_neighbors]
        i2_i3_n = [x for x in i2_neighbors if x in i3_neighbors]

        # Test i1-only neighbors
        for n in [x for x in i1_neighbors if x not in i1_i2_n and x not in i1_i3_n]:
            self.assertEqual(graph.nodes[n]['label'], 'E')
            self.assertEqual(3, len(get_neighbors_at(graph, n, graph.nodes[i1]['layer'])))

        # Test i2-only neighbors
        for n in [x for x in i2_neighbors if x not in i1_i2_n and x not in i2_i3_n]:
            self.assertEqual(graph.nodes[n]['label'], 'E')
            self.assertEqual(3, len(get_neighbors_at(graph, n, graph.nodes[i2]['layer'])))

        # Test i3-only neighbors
        for n in [x for x in i3_neighbors if x not in i1_i3_n and x not in i2_i3_n]:
            self.assertEqual(graph.nodes[n]['label'], 'E')
            self.assertEqual(3, len(get_neighbors_at(graph, n, graph.nodes[i3]['layer'])))

        # Test nodes connected to 2 interiors
        for n in [x for x in i1_i2_n if x not in i1_i3_n and x not in i2_i3_n]:
            self.assertEqual(graph.nodes[n]['label'], 'E')
            self.assertEqual(5, len(get_neighbors_at(graph, n, graph.nodes[i1]['layer'])))

        for n in [x for x in i1_i3_n if x not in i1_i2_n and x not in i2_i3_n]:
            self.assertEqual(graph.nodes[n]['label'], 'E')
            self.assertEqual(5, len(get_neighbors_at(graph, n, graph.nodes[i1]['layer'])))

        for n in [x for x in i2_i3_n if x not in i1_i2_n and x not in i1_i3_n]:
            self.assertEqual(graph.nodes[n]['label'], 'E')
            self.assertEqual(5, len(get_neighbors_at(graph, n, graph.nodes[i1]['layer'])))

        # Test nodes connected to 3 interiors
        for n in [x for x in i2_i3_n if x in i1_i2_n and x in i1_i3_n]:
            self.assertEqual(graph.nodes[n]['label'], 'E')
            self.assertEqual(7, len(get_neighbors_at(graph, n, graph.nodes[i1]['layer'])))

        if visualize_tests:
            visualize_graph_3d(graph)
            pyplot.show()

    def test_incorrect_label(self):
        graph = Graph()
        e1, e2, e3, e4, e5 = [gen_name() for _ in range(5)]

        graph.add_node(e1, layer=1, position=(1.0, 2.0), label='E')
        graph.add_node(e2, layer=1, position=(1.0, 1.5), label='E')
        graph.add_node(e3, layer=1, position=(1.0, 1.0), label='E')
        graph.add_node(e4, layer=1, position=(2.0, 1.0), label='E')
        graph.add_node(e5, layer=1, position=(1.5, 1.5), label='E')

        graph.add_edge(e1, e2)
        graph.add_edge(e2, e3)
        graph.add_edge(e3, e4)
        graph.add_edge(e4, e5)
        graph.add_edge(e5, e1)

        i = add_interior(graph, e1, e3, e4)
        graph.nodes[i]['label'] = 'i'

        if visualize_tests:
            visualize_graph_3d(graph)
            pyplot.show()

        with self.assertRaises(AssertionError):
            P4().apply(graph, [i])

        if visualize_tests:
            visualize_graph_3d(graph)
            pyplot.show()

    def test_missing_edges(self):
        graph = Graph()
        e1, e2, e3, e4, e5 = [gen_name() for _ in range(5)]

        graph.add_node(e1, layer=1, position=(1.0, 2.0), label='E')
        graph.add_node(e2, layer=1, position=(1.0, 1.5), label='E')
        graph.add_node(e3, layer=1, position=(1.0, 1.0), label='E')
        graph.add_node(e4, layer=1, position=(2.0, 1.0), label='E')
        graph.add_node(e5, layer=1, position=(1.5, 1.5), label='E')

        graph.add_edge(e1, e2)
        graph.add_edge(e3, e4)
        graph.add_edge(e5, e1)

        i = add_interior(graph, e1, e3, e4)

        if visualize_tests:
            visualize_graph_3d(graph)
            pyplot.show()

        with self.assertRaises(AssertionError):
            P4().apply(graph, [i])

        if visualize_tests:
            visualize_graph_3d(graph)
            pyplot.show()

    def test_incorrect_node_label(self):
        graph = Graph()
        e1, e2, e3, e4, e5 = [gen_name() for _ in range(5)]

        graph.add_node(e1, layer=1, position=(1.0, 2.0), label='E')
        graph.add_node(e2, layer=1, position=(1.0, 1.5), label='A')
        graph.add_node(e3, layer=1, position=(1.0, 1.0), label='D')
        graph.add_node(e4, layer=1, position=(2.0, 1.0), label='G')
        graph.add_node(e5, layer=1, position=(1.5, 1.5), label='B')

        graph.add_edge(e1, e2)
        graph.add_edge(e2, e3)
        graph.add_edge(e3, e4)
        graph.add_edge(e4, e5)
        graph.add_edge(e5, e1)

        i = add_interior(graph, e1, e3, e4)

        if visualize_tests:
            visualize_graph_3d(graph)
            pyplot.show()

        with self.assertRaises(AssertionError):
            P4().apply(graph, [i])

        if visualize_tests:
            visualize_graph_3d(graph)
            pyplot.show()

    def test_incorrect_interior(self):
        graph = Graph()
        e1, e2, e3, e4, e5 = [gen_name() for _ in range(5)]

        graph.add_node(e1, layer=1, position=(1.0, 2.0), label='I')
        graph.add_node(e2, layer=1, position=(1.0, 1.5), label='E')
        graph.add_node(e3, layer=1, position=(1.0, 1.0), label='E')
        graph.add_node(e4, layer=1, position=(2.0, 1.0), label='E')
        graph.add_node(e5, layer=1, position=(1.5, 1.5), label='E')

        graph.add_edge(e1, e2)
        graph.add_edge(e2, e3)
        graph.add_edge(e3, e4)
        graph.add_edge(e4, e5)
        graph.add_edge(e5, e1)

        if visualize_tests:
            visualize_graph_3d(graph)
            pyplot.show()

        with self.assertRaises(AssertionError):
            P4().apply(graph, [e1])

        if visualize_tests:
            visualize_graph_3d(graph)
            pyplot.show()

    def test_risky_triangle_break(self):
        graph = Graph()
        e1, e2, e3, e4, e5 = [gen_name() for _ in range(5)]
        e6 = gen_name()

        graph.add_node(e1, layer=1, position=(1.0, 2.0), label='E')
        graph.add_node(e2, layer=1, position=(1.0, 1.5), label='E')
        graph.add_node(e3, layer=1, position=(1.0, 1.0), label='E')
        graph.add_node(e4, layer=1, position=(2.0, 1.0), label='E')
        graph.add_node(e5, layer=1, position=(1.5, 1.5), label='E')
        graph.add_node(e6, layer=1, position=(0.0, 1.5), label='E')

        graph.add_edge(e1, e2)
        graph.add_edge(e2, e3)
        graph.add_edge(e3, e4)
        graph.add_edge(e4, e5)
        graph.add_edge(e5, e1)

        graph.add_edge(e1, e6)
        graph.add_edge(e6, e3)

        i = add_interior(graph, e1, e3, e4)

        if visualize_tests:
            visualize_graph_3d(graph)
            pyplot.show()

        [i1, i2, i3] = P4().apply(graph, [i])

        self.assertEqual(len(graph.nodes()), 15)
        self.assertEqual(len(graph.edges()), 29)

        self.assertEqual(graph.nodes[i]['label'], 'i')
        self.assertTrue(graph.has_edge(i, i1))
        self.assertTrue(graph.has_edge(i, i2))
        self.assertTrue(graph.has_edge(i, i3))

        self.assertEqual(graph.nodes[i1]['label'], 'I')
        self.assertEqual(graph.nodes[i2]['label'], 'I')
        self.assertEqual(graph.nodes[i3]['label'], 'I')
        self.assertEqual(graph.nodes[i1]['layer'], graph.nodes[i]['layer'] + 1)
        self.assertEqual(graph.nodes[i2]['layer'], graph.nodes[i]['layer'] + 1)
        self.assertEqual(graph.nodes[i3]['layer'], graph.nodes[i]['layer'] + 1)

        i1_neighbors = get_neighbors_at(graph, i1, graph.nodes[i1]['layer'])
        self.assertEqual(len(i1_neighbors), 3)
        i2_neighbors = get_neighbors_at(graph, i2, graph.nodes[i2]['layer'])
        self.assertEqual(len(i2_neighbors), 3)
        i3_neighbors = get_neighbors_at(graph, i3, graph.nodes[i3]['layer'])
        self.assertEqual(len(i3_neighbors), 3)

        i1_i2_n = [x for x in i1_neighbors if x in i2_neighbors]
        i1_i3_n = [x for x in i1_neighbors if x in i3_neighbors]
        i2_i3_n = [x for x in i2_neighbors if x in i3_neighbors]

        # Test i1-only neighbors
        for n in [x for x in i1_neighbors if x not in i1_i2_n and x not in i1_i3_n]:
            self.assertEqual(graph.nodes[n]['label'], 'E')
            self.assertEqual(3, len(get_neighbors_at(graph, n, graph.nodes[i1]['layer'])))

        # Test i2-only neighbors
        for n in [x for x in i2_neighbors if x not in i1_i2_n and x not in i2_i3_n]:
            self.assertEqual(graph.nodes[n]['label'], 'E')
            self.assertEqual(3, len(get_neighbors_at(graph, n, graph.nodes[i2]['layer'])))

        # Test i3-only neighbors
        for n in [x for x in i3_neighbors if x not in i1_i3_n and x not in i2_i3_n]:
            self.assertEqual(graph.nodes[n]['label'], 'E')
            self.assertEqual(3, len(get_neighbors_at(graph, n, graph.nodes[i3]['layer'])))

        # Test nodes connected to 2 interiors
        for n in [x for x in i1_i2_n if x not in i1_i3_n and x not in i2_i3_n]:
            self.assertEqual(graph.nodes[n]['label'], 'E')
            self.assertEqual(5, len(get_neighbors_at(graph, n, graph.nodes[i1]['layer'])))

        for n in [x for x in i1_i3_n if x not in i1_i2_n and x not in i2_i3_n]:
            self.assertEqual(graph.nodes[n]['label'], 'E')
            self.assertEqual(5, len(get_neighbors_at(graph, n, graph.nodes[i1]['layer'])))

        for n in [x for x in i2_i3_n if x not in i1_i2_n and x not in i1_i3_n]:
            self.assertEqual(graph.nodes[n]['label'], 'E')
            self.assertEqual(5, len(get_neighbors_at(graph, n, graph.nodes[i1]['layer'])))

        # Test nodes connected to 3 interiors
        for n in [x for x in i2_i3_n if x in i1_i2_n and x in i1_i3_n]:
            self.assertEqual(graph.nodes[n]['label'], 'E')
            self.assertEqual(7, len(get_neighbors_at(graph, n, graph.nodes[i1]['layer'])))

        if visualize_tests:
            visualize_graph_3d(graph)
            pyplot.show()

    def test_in_bigger_graph(self):
        graph = Graph()
        positions = [
            (0, 0),
            (0, 1),
            (1, 0),
            (0, 0.5),
            (0.5, 0.5)
        ]

        vx_e1 = gen_name()
        vx_e2 = gen_name()
        vx_e3 = gen_name()
        vx_e12 = gen_name()
        vx_e23 = gen_name()
        graph.add_node(vx_e1, layer=0, position=positions[0], label='E')
        graph.add_node(vx_e2, layer=0, position=positions[1], label='E')
        graph.add_node(vx_e3, layer=0, position=positions[2], label='E')
        graph.add_node(vx_e12, layer=0, position=positions[3], label='E')
        graph.add_node(vx_e23, layer=0, position=positions[4], label='E')

        graph.add_edge(vx_e1, vx_e12)
        graph.add_edge(vx_e12, vx_e2)
        graph.add_edge(vx_e2, vx_e23)
        graph.add_edge(vx_e23, vx_e3)
        graph.add_edge(vx_e3, vx_e1)

        vx_e1122 = gen_name()
        graph.add_node(vx_e1122, layer=0, position=(-0.5, 0.5), label='E')
        graph.add_edge(vx_e1, vx_e1122)
        graph.add_edge(vx_e12, vx_e1122)
        graph.add_edge(vx_e2, vx_e1122)

        vx_e2233 = gen_name()
        graph.add_node(vx_e2233, layer=0, position=(1, 1), label='E')
        graph.add_edge(vx_e2, vx_e2233)
        graph.add_edge(vx_e23, vx_e2233)
        graph.add_edge(vx_e3, vx_e2233)

        vx_e13 = gen_name()
        graph.add_node(vx_e13, layer=0, position=(0.5, -0.5), label='E')
        graph.add_edge(vx_e1, vx_e13)
        graph.add_edge(vx_e3, vx_e13)

        I = add_interior(graph, vx_e1, vx_e2, vx_e3)
        I1 = add_interior(graph, vx_e1122, vx_e1, vx_e12)
        I2 = add_interior(graph, vx_e1122, vx_e12, vx_e2)
        I3 = add_interior(graph, vx_e2233, vx_e2, vx_e23)
        I4 = add_interior(graph, vx_e2233, vx_e23, vx_e3)
        I5 = add_interior(graph, vx_e1, vx_e3, vx_e13)

        if visualize_tests:
            visualize_graph_3d(graph)
            pyplot.show()

        [i1, i2, i3] = P4().apply(graph, [I])

        self.assertEqual(len(graph.nodes()), 22)
        self.assertEqual(len(graph.edges()), 50)

        self.assertEqual(graph.nodes[I]['label'], 'i')
        self.assertTrue(graph.has_edge(I, i1))
        self.assertTrue(graph.has_edge(I, i2))
        self.assertTrue(graph.has_edge(I, i3))

        self.assertEqual(graph.nodes[i1]['label'], 'I')
        self.assertEqual(graph.nodes[i2]['label'], 'I')
        self.assertEqual(graph.nodes[i3]['label'], 'I')
        self.assertEqual(graph.nodes[i1]['layer'], graph.nodes[I]['layer'] + 1)
        self.assertEqual(graph.nodes[i2]['layer'], graph.nodes[I]['layer'] + 1)
        self.assertEqual(graph.nodes[i3]['layer'], graph.nodes[I]['layer'] + 1)

        i1_neighbors = get_neighbors_at(graph, i1, graph.nodes[i1]['layer'])
        self.assertEqual(len(i1_neighbors), 3)
        i2_neighbors = get_neighbors_at(graph, i2, graph.nodes[i2]['layer'])
        self.assertEqual(len(i2_neighbors), 3)
        i3_neighbors = get_neighbors_at(graph, i3, graph.nodes[i3]['layer'])
        self.assertEqual(len(i3_neighbors), 3)

        i1_i2_n = [x for x in i1_neighbors if x in i2_neighbors]
        i1_i3_n = [x for x in i1_neighbors if x in i3_neighbors]
        i2_i3_n = [x for x in i2_neighbors if x in i3_neighbors]

        # Test i1-only neighbors
        for n in [x for x in i1_neighbors if x not in i1_i2_n and x not in i1_i3_n]:
            self.assertEqual(graph.nodes[n]['label'], 'E')
            self.assertEqual(3, len(get_neighbors_at(graph, n, graph.nodes[i1]['layer'])))

        # Test i2-only neighbors
        for n in [x for x in i2_neighbors if x not in i1_i2_n and x not in i2_i3_n]:
            self.assertEqual(graph.nodes[n]['label'], 'E')
            self.assertEqual(3, len(get_neighbors_at(graph, n, graph.nodes[i2]['layer'])))

        # Test i3-only neighbors
        for n in [x for x in i3_neighbors if x not in i1_i3_n and x not in i2_i3_n]:
            self.assertEqual(graph.nodes[n]['label'], 'E')
            self.assertEqual(3, len(get_neighbors_at(graph, n, graph.nodes[i3]['layer'])))

        # Test nodes connected to 2 interiors
        for n in [x for x in i1_i2_n if x not in i1_i3_n and x not in i2_i3_n]:
            self.assertEqual(graph.nodes[n]['label'], 'E')
            self.assertEqual(5, len(get_neighbors_at(graph, n, graph.nodes[i1]['layer'])))

        for n in [x for x in i1_i3_n if x not in i1_i2_n and x not in i2_i3_n]:
            self.assertEqual(graph.nodes[n]['label'], 'E')
            self.assertEqual(5, len(get_neighbors_at(graph, n, graph.nodes[i1]['layer'])))

        for n in [x for x in i2_i3_n if x not in i1_i2_n and x not in i1_i3_n]:
            self.assertEqual(graph.nodes[n]['label'], 'E')
            self.assertEqual(5, len(get_neighbors_at(graph, n, graph.nodes[i1]['layer'])))

        # Test nodes connected to 3 interiors
        for n in [x for x in i2_i3_n if x in i1_i2_n and x in i1_i3_n]:
            self.assertEqual(graph.nodes[n]['label'], 'E')
            self.assertEqual(7, len(get_neighbors_at(graph, n, graph.nodes[i1]['layer'])))

        if visualize_tests:
            visualize_graph_3d(graph)
            pyplot.show()
