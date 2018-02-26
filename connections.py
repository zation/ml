class Connections:
    def __init__(self):
        self.connections = []

    def append_connection(self, connection):
        self.connections.append(connection)

    def dump(self):
        for connection in self.connections:
            print(connection)
