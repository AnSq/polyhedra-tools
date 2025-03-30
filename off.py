#!/usr/bin/env python

from typing import Any, TypeVar, TYPE_CHECKING
import warnings
import logging
import itertools
from math import sqrt, log10

from skspatial.objects import Points as sksoPoints  #todo: remove this as a dependancy

if TYPE_CHECKING:
    import _typeshed


logger = logging.getLogger(__name__)

TOLERANCE = 1e-12
ROUND_DIGITS = int(-log10(TOLERANCE))


def passfail(x:Any):
    return "PASS" if x else "FAIL"


T = TypeVar("T", bound="_typeshed.SupportsRichComparison")
def _make_edges_from_list(points:list[T]) -> list[tuple[T,T]]:
    return [(min(x), max(x)) for x in zip(points, points[1:]+[points[0]])]



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


    def __eq__(self, other:Any) -> bool:
        if isinstance(other, self.__class__):
            # equal if their locations are equal. index doesn't matter
            return self.coords == other.coords
        else:
            return False


    def __gt__(self, other:Any):
        if isinstance(other, self.__class__):
            return self.index > other.index
        else:
            return False


    __ge__ = None



class Edge:
    def __init__(self, p0:Vertex, p1:Vertex) -> None:
        self.p0 = p0
        self.p1 = p1


    def __str__(self) -> str:
        return f"Edge(v{self.p0.index}, v{self.p1.index})"


    def __repr__(self) -> str:
        return str(self)


    def __eq__(self, other:Any) -> bool:
        if isinstance(other, self.__class__):
            # equal if their points are equal. direction doesn't matter
            return (self.p0 == other.p0 and self.p1 == other.p1) or (self.p0 == other.p1 and self.p1 == other.p0)
        else:
            return False


    @property
    def length(self) -> float:
        p0 = self.p0
        p1 = self.p1
        return sqrt((p1.x - p0.x)**2 + (p1.y - p0.y)**2 + (p1.z - p0.z)**2)



class Face:
    def __init__(self, index:int, points:list[Vertex]) -> None:
        self.index = index
        self.points = points
        self._normal = None


    @property
    def edge_indexes(self) -> list[tuple[int,int]]:
        """a list of point index tuples (p0.index, p1.index) for each edge of this face"""
        return _make_edges_from_list([x.index for x in self.points])


    @property
    def edges(self) -> list[Edge]:
        return [Edge(e[0], e[1]) for e in _make_edges_from_list(self.points)]


    def __str__(self) -> str:
        return f"<{len(self.points)}-gon f{self.index} {[i.index for i in self.points]}>"


    def __repr__(self) -> str:
        return str(self)


    def points_array(self) -> list[list[float]]:
        return [p.coords for p in self.points]


    def check_coplanar(self, tol=TOLERANCE):
        result = sksoPoints(self.points_array()).are_coplanar(tol=tol)
        if not result:
            print(f"face {self} is not coplanar")
        return result


    @property
    def normal(self):
        if self._normal is None:
            return self.calculate_normal()
        return self._normal


    def calculate_normal(self):
        """normally you can just use the Face.normal property which uses a cached version, but this is here if you want to explicity refresh it"""
        self._normal = None #TODO
        return self._normal



class Mesh:
    def __init__(self, points:list[Vertex], faces:list[Face], edges:list[Edge], _incorrect_edges=False, _unfinished_read=False) -> None:
        self.points = points
        self.faces = faces
        self.edges = edges
        self._incorrect_edges = _incorrect_edges
        self._unfinished_read = _unfinished_read


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
            if count != len(point_indexes):
                raise ValueError(f"face #{f} declared with {count} points but it appears to have {len(point_indexes)}")
            face_edges = set(_make_edges_from_list(point_indexes))
            edge_point_indexes |= face_edges
            faces.append(Face(f, [vertices[i] for i in point_indexes]))

        edges:list[Edge] = []
        for e in edge_point_indexes:
            edges.append(Edge(vertices[e[0]], vertices[e[1]]))

        _incorrect_edges = False
        if nedges != len(edges):
            warnings.warn(f"[{info}] declared edges {nedges} != actual edges {len(edges)}")
            _incorrect_edges = True

        _unfinished_read = False
        try:
            next(lines)
        except StopIteration:
            pass
        else:
            warnings.warn(f"[{info}] file not finished reading???")
            _unfinished_read = True

        return cls(vertices, faces, edges, _incorrect_edges, _unfinished_read)


    @classmethod
    def load(cls, fname, *, info=False):
        with open(fname) as f:
            return cls.loads(f.read(), info=fname if info else "")


    @property
    def num_points(self):
        return len(self.points)


    @property
    def num_edges(self):
        return len(self.edges)


    @property
    def num_faces(self):
        return len(self.faces)


    def get_point_face_use_counts(self) -> list[int]:
        """get the number of times each point is used in a face"""
        used = [0] * self.num_points
        for face in self.faces:
            for point in face.points:
                used[point.index] += 1
        return used


    def get_point_edge_use_counts(self) -> list[int]:
        """get the number of times each point is used in an edge"""
        used = [0] * self.num_points
        for edge in self.edges:
            used[edge.p0.index] += 1
            used[edge.p1.index] += 1
        return used


    def get_unused_points(self) -> list[int]:
        used = self.get_point_edge_use_counts()
        unused:list[int] = []
        for i in range(len(used)):
            if not used[i]:
                unused.append(i)
        return unused

    def points_array(self) -> list[list[float]]:
        return [p.coords for p in self.points]


    def check_no_duplicate_points(self):
        result = True
        for v0, v1 in itertools.combinations(self.points, 2):
            if v0 == v1:
                print(f"duplicate points {v0} {v1}")
                result = False
        return result


    def check_no_unused_points(self):
        unused = self.get_unused_points()
        if unused:
            print(f"unused points: {unused}")
        return not unused


    def check_no_duplicate_edges(self):
        result = True
        for e0, e1 in itertools.combinations(self.edges, 2):
            if e0 == e1:
                print(f"duplicate edges {e0} {e1}")
                result = False
        return result


    def check_closed_points(self) -> bool:
        """check that the mesh is closed by checking that each point is used by the same number of faces as edges"""
        return self.get_point_face_use_counts() == self.get_point_edge_use_counts()


    def check_closed_edges(self) -> bool:
        """check that the mesh is closed by checking that each edge is used by exactly 2 faces"""
        edge_uses:dict[tuple[int,int],int] = {}
        for face in self.faces:
            for e in face.edge_indexes:
                if e not in edge_uses:
                    edge_uses[e] = 0
                edge_uses[e] += 1

        for edge in self.edges:
            e = (edge.p0.index, edge.p1.index)
            if e not in edge_uses:
                    edge_uses[e] = 0

        result = True
        for eu in edge_uses:
            if edge_uses[eu] != 2:
                result = False
                print(f"edge {eu} is used {edge_uses[eu]} times")
        return result


    def check_unit_edges(self, round_digits=ROUND_DIGITS) -> bool:
        """check that all edges have a length of 1"""
        lengths = set(round(e.length, round_digits) for e in self.edges)
        if lengths != {1}:
            logger.error(f"edge lengths: {sorted(list(lengths))}")
            return False
        return True


    def all_checks(self, include_unit_edges=False) -> bool:
        print("Checking mesh")
        ndp = self.check_no_duplicate_points()
        nup = self.check_no_unused_points()
        nde = self.check_no_duplicate_edges()
        ccp = self.check_closed_points()
        cce = self.check_closed_edges()
        coplanar_faces = [f.check_coplanar() for f in self.faces]
        cpf = all(coplanar_faces)
        iec = not self._incorrect_edges
        ufr = not self._unfinished_read
        print(f"No duplicate points:  {passfail(ndp)}")
        print(f"No unused points:     {passfail(nup)}")
        print(f"No duplicate edges:   {passfail(nde)}")
        print(f"Closed points:        {passfail(ccp)}")
        print(f"Closed edges:         {passfail(cce)}")
        print(f"Coplanar faces:       {passfail(cpf)}")
        print(f"Incorrect edge count: {passfail(iec)}")
        print(f"Unfinished file read: {passfail(ufr)}")

        uel = True
        if include_unit_edges:
            uel = self.check_unit_edges()
            print(f"Unit edge lengths:    {passfail(uel)}")

        return all((ndp, nup, nde, ccp, cce, cpf, iec, ufr, uel))


if __name__ == "__main__":
    import sys, os
    f = sys.argv[1]
    include_unit_edges = len(sys.argv) > 2 and sys.argv[2] == "-e"
    result = True
    if os.path.isdir(f):
        for fname in os.listdir(f):
            fullname = os.path.join(f, fname)
            if fullname.endswith(".off") and os.path.isfile(fullname):
                print(f"\n{fname}")
                mesh = Mesh.load(fullname)
                result = all((mesh.all_checks(include_unit_edges), result))
    elif os.path.isfile(f):
        mesh = Mesh.load(f)
        result = mesh.all_checks(include_unit_edges)
    else:
        print(f"Invalid input: {f}")
        result = False

    if result:
        print("\nAll checks PASSED")
    else:
        print("\nSome checks FAILED")
