import numpy as np


"""
The lib for the Wiring is taken from https://github.com/mlech26l/ncps.
"""


class Wiring:
    def __init__(self, units):
        self.units = units
        self.adjacency_matrix = np.zeros([units, units], dtype=np.int32)
        self.sensory_adjacency_matrix = None
        self.input_dim = None
        self.output_dim = None

    @property
    def num_layers(self):
        return 1

    def get_neurons_of_layer(self, layer_id):
        return list(range(self.units))

    def is_built(self):
        return self.input_dim is not None

    def build(self, input_dim):
        if not self.input_dim is None and self.input_dim != input_dim:
            raise ValueError(
                "Conflicting input dimensions provided. set_input_dim() was called with {} but actual input has dimension {}".format(
                    self.input_dim, input_dim
                )
            )
        if self.input_dim is None:
            self.set_input_dim(input_dim)

    def erev_initializer(self, shape=None, dtype=None):
        return np.copy(self.adjacency_matrix)

    def sensory_erev_initializer(self, shape=None, dtype=None):
        return np.copy(self.sensory_adjacency_matrix)

    def set_input_dim(self, input_dim):
        self.input_dim = input_dim
        self.sensory_adjacency_matrix = np.zeros(
            [input_dim, self.units], dtype=np.int32
        )

    def set_output_dim(self, output_dim):
        self.output_dim = output_dim

    # May be overwritten by child class
    def get_type_of_neuron(self, neuron_id):
        return "motor" if neuron_id < self.output_dim else "inter"

    def add_synapse(self, src, dest, polarity):
        if src < 0 or src >= self.units:
            raise ValueError(
                "Cannot add synapse originating in {} if cell has only {} units".format(
                    src, self.units
                )
            )
        if dest < 0 or dest >= self.units:
            raise ValueError(
                "Cannot add synapse feeding into {} if cell has only {} units".format(
                    dest, self.units
                )
            )
        if not polarity in [-1, 1]:
            raise ValueError(
                "Cannot add synapse with polarity {} (expected -1 or +1)".format(
                    polarity
                )
            )
        self.adjacency_matrix[src, dest] = polarity

    def add_sensory_synapse(self, src, dest, polarity):
        if self.input_dim is None:
            raise ValueError(
                "Cannot add sensory synapses before build() has been called!"
            )
        if src < 0 or src >= self.input_dim:
            raise ValueError(
                "Cannot add sensory synapse originating in {} if input has only {} features".format(
                    src, self.input_dim
                )
            )
        if dest < 0 or dest >= self.units:
            raise ValueError(
                "Cannot add synapse feeding into {} if cell has only {} units".format(
                    dest, self.units
                )
            )
        if not polarity in [-1, 1]:
            raise ValueError(
                "Cannot add synapse with polarity {} (expected -1 or +1)".format(
                    polarity
                )
            )
        self.sensory_adjacency_matrix[src, dest] = polarity

    def get_config(self):
        return {
            "adjacency_matrix": self.adjacency_matrix,
            "sensory_adjacency_matrix": self.sensory_adjacency_matrix,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "units": self.units,
        }

    @classmethod
    def from_config(cls, config):
        # There might be a cleaner solution but it will work
        wiring = Wiring(config["units"])
        wiring.adjacency_matrix = config["adjacency_matrix"]
        wiring.sensory_adjacency_matrix = config["sensory_adjacency_matrix"]
        wiring.input_dim = config["input_dim"]
        wiring.output_dim = config["output_dim"]

        return wiring

    def get_graph(self, include_sensory_neurons=True):
        if not self.is_built():
            raise ValueError(
                "Wiring is not built yet.\n"
                "This is probably because the input shape is not known yet.\n"
                "Consider calling the model.build(...) method using the shape of the inputs."
            )
        # Only import networkx package if we really need it
        import networkx as nx

        DG = nx.DiGraph()
        for i in range(self.units):
            neuron_type = self.get_type_of_neuron(i)
            DG.add_node("neuron_{:d}".format(i), neuron_type=neuron_type)
        for i in range(self.input_dim):
            DG.add_node("sensory_{:d}".format(i), neuron_type="sensory")

        erev = self.adjacency_matrix
        sensory_erev = self.sensory_adjacency_matrix

        for src in range(self.input_dim):
            for dest in range(self.units):
                if self.sensory_adjacency_matrix[src, dest] != 0:
                    polarity = (
                        "excitatory" if sensory_erev[src, dest] >= 0.0 else "inhibitory"
                    )
                    DG.add_edge(
                        "sensory_{:d}".format(src),
                        "neuron_{:d}".format(dest),
                        polarity=polarity,
                    )

        for src in range(self.units):
            for dest in range(self.units):
                if self.adjacency_matrix[src, dest] != 0:
                    polarity = "excitatory" if erev[src, dest] >= 0.0 else "inhibitory"
                    DG.add_edge(
                        "neuron_{:d}".format(src),
                        "neuron_{:d}".format(dest),
                        polarity=polarity,
                    )
        return DG

    @property
    def synapse_count(self):
        return np.sum(np.abs(self.adjacency_matrix))

    @property
    def sensory_synapse_count(self):
        return np.sum(np.abs(self.sensory_adjacency_matrix))

    def draw_graph(
        self,
        layout="shell",
        neuron_colors=None,
        synapse_colors=None,
        draw_labels=False,
    ):
        # May switch to Cytoscape once support in Google Colab is available
        # https://stackoverflow.com/questions/62421021/how-do-i-install-cytoscape-on-google-colab
        import networkx as nx
        import matplotlib.patches as mpatches

        if isinstance(synapse_colors, str):
            synapse_colors = {
                "excitatory": synapse_colors,
                "inhibitory": synapse_colors,
            }
        elif synapse_colors is None:
            synapse_colors = {"excitatory": "tab:green", "inhibitory": "tab:red"}

        default_colors = {
            "inter": "tab:blue",
            "motor": "tab:orange",
            "sensory": "tab:olive",
        }
        if neuron_colors is None:
            neuron_colors = {}
        # Merge default with user provided color dict
        for k, v in default_colors.items():
            if not k in neuron_colors.keys():
                neuron_colors[k] = v

        legend_patches = []
        for k, v in neuron_colors.items():
            label = "{}{} neurons".format(k[0].upper(), k[1:])
            color = v
            legend_patches.append(mpatches.Patch(color=color, label=label))

        G = self.get_graph()
        layouts = {
            "kamada": nx.kamada_kawai_layout,
            "circular": nx.circular_layout,
            "random": nx.random_layout,
            "shell": nx.shell_layout,
            "spring": nx.spring_layout,
            "spectral": nx.spectral_layout,
            "spiral": nx.spiral_layout,
        }
        if not layout in layouts.keys():
            raise ValueError(
                "Unknown layer '{}', use one of '{}'".format(
                    layout, str(layouts.keys())
                )
            )
        pos = layouts[layout](G)

        # Draw neurons
        for i in range(self.units):
            node_name = "neuron_{:d}".format(i)
            neuron_type = G.nodes[node_name]["neuron_type"]
            neuron_color = "tab:blue"
            if neuron_type in neuron_colors.keys():
                neuron_color = neuron_colors[neuron_type]
            nx.draw_networkx_nodes(G, pos, [node_name], node_color=neuron_color)

        # Draw sensory neurons
        for i in range(self.input_dim):
            node_name = "sensory_{:d}".format(i)
            neuron_color = "blue"
            if "sensory" in neuron_colors.keys():
                neuron_color = neuron_colors["sensory"]
            nx.draw_networkx_nodes(G, pos, [node_name], node_color=neuron_color)

        # Optional: draw labels
        if draw_labels:
            nx.draw_networkx_labels(G, pos)

        # Draw edges
        for node1, node2, data in G.edges(data=True):
            polarity = data["polarity"]
            edge_color = synapse_colors[polarity]
            nx.draw_networkx_edges(G, pos, [(node1, node2)], edge_color=edge_color)

        return legend_patches


class FullyConnected(Wiring):
    def __init__(
        self, units, output_dim=None, erev_init_seed=1111, self_connections=True
    ):
        super(FullyConnected, self).__init__(units)
        if output_dim is None:
            output_dim = units
        self.self_connections = self_connections
        self.set_output_dim(output_dim)
        self._rng = np.random.default_rng(erev_init_seed)
        for src in range(self.units):
            for dest in range(self.units):
                if src == dest and not self_connections:
                    continue
                polarity = self._rng.choice([-1, 1, 1])
                self.add_synapse(src, dest, polarity)

    def build(self, input_shape):
        super().build(input_shape)
        for src in range(self.input_dim):
            for dest in range(self.units):
                polarity = self._rng.choice([-1, 1, 1])
                self.add_sensory_synapse(src, dest, polarity)


class Random(Wiring):
    def __init__(self, units, output_dim=None, sparsity_level=0.0, random_seed=1111):
        super(Random, self).__init__(units)
        if output_dim is None:
            output_dim = units
        self.set_output_dim(output_dim)
        self.sparsity_level = sparsity_level

        if sparsity_level < 0.0 or sparsity_level >= 1.0:
            raise ValueError(
                "Invalid sparsity level '{}', expected value in range [0,1)".format(
                    sparsity_level
                )
            )
        self._rng = np.random.default_rng(random_seed)

        number_of_synapses = int(np.round(units * units * (1 - sparsity_level)))
        all_synapses = []
        for src in range(self.units):
            for dest in range(self.units):
                all_synapses.append((src, dest))

        # noinspection PyTypeChecker
        used_synapses = self._rng.choice(
            all_synapses, size=number_of_synapses, replace=False
        )
        for src, dest in used_synapses:
            polarity = self._rng.choice([-1, 1, 1])
            self.add_synapse(src, dest, polarity)

    def build(self, input_shape):
        super().build(input_shape)
        number_of_sensory_synapses = int(
            np.round(self.input_dim * self.units * (1 - self.sparsity_level))
        )
        all_sensory_synapses = []
        for src in range(self.input_dim):
            for dest in range(self.units):
                all_sensory_synapses.append((src, dest))

        # noinspection PyTypeChecker
        used_sensory_synapses = self._rng.choice(
            all_sensory_synapses, size=number_of_sensory_synapses, replace=False
        )
        for src, dest in used_sensory_synapses:
            polarity = self._rng.choice([-1, 1, 1])
            self.add_sensory_synapse(src, dest, polarity)
            polarity = self._rng.choice([-1, 1, 1])
            self.add_sensory_synapse(src, dest, polarity)