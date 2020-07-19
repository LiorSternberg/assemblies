import matplotlib.pyplot as plt
from networkx import DiGraph, has_path, draw, draw_networkx_edge_labels, get_node_attributes, get_edge_attributes

from learning.components.errors import NoPathException, \
    IllegalOutputAreasException


class ConnectionsGraph(DiGraph):
    """
    A graph representing the connections between source and target elements of
    projections in a sequence. The connections graph is used to validate the
    flow of the sequence (ensure inputs are connected to the output), and to
    display the sequence, allowing users to review the sequence of projections
    they defined.
    """
    def add_connection(self, source_node, target_node, weight, number_of_iterations):
        """
        Adding a connection to the ConnectionsGraph by adding an edge between
        the source and target nodes (create them if needed).
        """
        existing_horizontal_positions = max([position[1] for position in get_node_attributes(
            self, 'position').values()]) if self.nodes else 0
        horizontal_index = min(number_of_iterations, existing_horizontal_positions)

        if not self.has_node(source_node):
            vertical_index = len([position for position in get_node_attributes(
                self, 'position').values() if position[1] == horizontal_index])
            self.add_node(source_node, position=(vertical_index, horizontal_index))

        if not self.has_node(target_node):
            vertical_index = len([position for position in get_node_attributes(
                self, 'position').values() if position[1] == horizontal_index + 1])
            self.add_node(target_node, position=(vertical_index, horizontal_index + 1))

        if self.has_edge(source_node, target_node):
            # Adding to the previous edge weight
            weight += self.get_edge_data(source_node, target_node)['weight']

        self.add_edge(source_node, target_node, weight=weight)

    def verify_inputs_are_connected_to_output(self):
        """
        Checking that there is a directed path of projection between each
        input (either stimulus or input_bit) and each output area.
        In case one doesn't exist, this function raises an exception.

        Note that we only check stimuli which are in the connections graph,
        meaning only stimuli that function as inputs in the sequence are
        validated to have a path to the output area.
        """
        input_types = ('stimulus', 'input-bit')

        for input_type in input_types:
            input_nodes = [node for node in self.nodes
                           if node.startswith(input_type)]

            output_areas = [node for node in self.nodes if
                            node.startswith('output')]

            for input_node in input_nodes:
                for area in output_areas:
                    if not has_path(self, input_node, area):
                        raise NoPathException(input_node[len(input_type) + 1:],
                                              area[len('output') + 1:])

    def verify_single_output_area(self) -> str:
        """
        Checking that there is a single output area in the sequence.
        In any other case (none or multiple output areas), this function raises
         an exception.

        :return: The name of the output area in the connections graph.
        """
        output_areas = [node[7:] for node in self.nodes if node.startswith('output')]
        if len(output_areas) != 1:
            raise IllegalOutputAreasException(output_areas)
        return output_areas[0]

    def display(self):
        """
        Displaying the sequence's connections graph
        """
        node_positions = get_node_attributes(self, 'position')
        edge_labels = get_edge_attributes(self, 'weight')
        draw(self, pos=node_positions, alpha=1, with_labels=True)
        draw_networkx_edge_labels(self, node_positions, edge_labels=edge_labels)
        plt.show()
