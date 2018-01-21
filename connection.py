class Connection:
    def __init__(self, upsteam_node, downstream_node):
        self.upsteam_node = upsteam_node
        self.downstream_node = downstream_node
        self.weight = random.uniform(-0.1, 0.1)
        self.gradient = 0.0

    def calculate_gradient(self):
        self.gradient = self.downstream_node.delta * self.upsteam_node.output

    def get_gradient(self):
        return self.gradient

    def update_weight(self, weight):
        self.weight = weight

    def __str__(self):
        return '(%u - %u) -> (%u - %u) = %f' % (
            self.upstream_node.layer_index,
            self.upstream_node.node_index,
            self.downstream_node.layer_index,
            self.downstream_node.node_index,
            self.weight
        )
