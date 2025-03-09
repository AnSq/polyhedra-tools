#!/usr/bin/env python

class Vertex:
    def __init__(self, index:int, x:float, y:float, z:float) -> None:
        self.index = index
        self.coords = [x, y, z]

    @property
    def x(self):
        return self.coords[0]
    @x.setter
    def x(self, value:float):
        self.coords[0] = value

    @property
    def y(self):
        return self.coords[1]
    @y.setter
    def y(self, value:float):
        self.coords[1] = value

    @property
    def z(self):
        return self.coords[2]
    @z.setter
    def z(self, value:float):
        self.coords[2] = value

    def __str__(self) -> str:
        return f"<v{self.index} ({self.x}, {self.y}, {self.z})>"

    def __repr__(self) -> str:
        return str(self)


class Face:
    def __init__(self, index:int, points:list[Vertex]) -> None:
        self.index = index
        self.points = points
        self._normal = None

    def __str__(self) -> str:
        return f"<{len(self.points)}-gon f{self.index} {[i.index for i in self.points]}>"

    def __repr__(self) -> str:
        return str(self)

    def points_array(self) -> list[list[float]]:
        return [p.coords for p in self.points]

    @property
    def normal(self):
        if self._normal is None:
            return self.calculate_normal()
        return self._normal

    def calculate_normal(self):
        """normally you can just use the Face.normal property which uses a cached version, but this is here if you want to explicity refresh it"""
        self._normal = None #TODO
        return self._normal


class Edge:
    def __init__(self, p0:Vertex, p1:Vertex) -> None:
        self.p0 = p0
        self.p1 = p1

    def __str__(self) -> str:
        return f"Edge(v{self.p0.index}, v{self.p1.index})"

    def __repr__(self) -> str:
        return str(self)


class Mesh:
    def __init__(self, points:list[Vertex], faces:list[Face], edges:list[Edge]) -> None:
        self.points = points
        self.faces = faces
        self.edges = edges

    def __str__(self) -> str:
        return f"Mesh(v{len(self.points)}, f{len(self.faces)}, e{len(self.edges)})"

    def __repr__(self) -> str:
        return str(self)

    @classmethod
    def loads(cls, s:str, *, info=""):
        lines = (j for j in (i.split("#")[0].strip() for i in s.split("\n")) if j != "")

        line = next(lines)
        if line == "OFF":
            line = next(lines)
        nvertices, nfaces, nedges = (int(i) for i in line.split())

        vertices:list[Vertex] = []
        for v in range(nvertices):
            line = next(lines)
            x, y, z = (float(i) for i in line.split())
            vertices.append(Vertex(v, x, y, z))

        faces:list[Face] = []
        edge_point_indexes:set[tuple[int,int]] = set()
        for f in range(nfaces):
            line = next(lines)
            g = (int(i) for i in line.split())
            count = next(g)
            point_indexes = list(g)
            assert count == len(point_indexes)
            face_edges = set((min(x), max(x)) for x in zip(point_indexes, point_indexes[1:]+[point_indexes[0]]))
            edge_point_indexes |= face_edges
            faces.append(Face(f, [vertices[i] for i in point_indexes]))

        edges:list[Edge] = []
        for e in edge_point_indexes:
            edges.append(Edge(vertices[e[0]], vertices[e[1]]))

        if nedges != len(edges):
            print(f"[{info}] declared edges {nedges} != actual edges {len(edges)}")

        try:
            next(lines)
        except StopIteration:
            pass
        else:
            print(f"[{info}] file not finished reading???")

        return cls(vertices, faces, edges)


    @classmethod
    def load(cls, fname, *, info=False):
        with open(fname) as f:
            return cls.loads(f.read(), info=fname if info else "")


    def points_array(self) -> list[list[float]]:
        return [p.coords for p in self.points]


if __name__ == "__main__":
    import sys
    with open(sys.argv[1]) as f:
        s = f.read()
    Mesh.loads(s)
