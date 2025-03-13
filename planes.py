#!/usr/bin/env python
import os
import time
import itertools
import math
import shelve
import logging
import argparse

from typing import TypeAlias
from collections.abc import Sequence

from skspatial.objects import Plane, Point, Points, LineSegment, Vector

import numpy as np
from SetCoverPy import setcover  #type: ignore

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D  #type: ignore
import mpl_toolkits.mplot3d.art3d as art3d  #type: ignore

import off

PlaneSet:TypeAlias = set[frozenset[int]]

logging.captureWarnings(True)

TOLERANCE = 1e-12
DB_FILE = "database.db"


def open_db(fname=DB_FILE) -> shelve.Shelf:
    return shelve.open(fname, writeback=True)


def load_mesh(db:shelve.Shelf, fname:str, clear=False) -> tuple[str,off.Mesh]:
    """load the given mesh file into the db. Deletes the entire db entry (planes, solutions, etc) for the mesh first if clear=True"""
    name = os.path.basename(fname).split("_")[0]

    if clear:
        del db[name]

    mesh = off.Mesh.load(fname, info=True)

    if name not in db:
        ub, reason = find_upper_bound(mesh)
        db[name] = {
            "upper_bound": ub,
            "ub_reason": reason,
            "ub_reason": reason,
            "best_solution": None,
            "num_solutions":  0,
            "solutions": set(),
            "planes": []
        }

    db[name]["mesh"] = mesh

    return (name, mesh)


def load_meshes(db:shelve.Shelf, folder:str, clear=False):
    """loads all .off mesh files in the given directory into the db"""
    for fname in os.listdir(folder):
        fullname = os.path.join(folder, fname)
        if fullname.endswith(".off") and os.path.isfile(fullname):
            load_mesh(db, fullname, clear)
    db.sync()


def find_planes(mesh:off.Mesh, name:str=None) -> PlaneSet:
    """find all sets of coplanar vertices for a mesh"""

    f_errors = {}
    for f in mesh.faces:
        if len(f.points) > 3:
            points = Points(f.points_array())
            if not points.are_coplanar(tol=TOLERANCE):
                plane = Plane.from_points(f.points[0].coords, f.points[1].coords, f.points[2].coords)
                distances = []
                for p in f.points[3:]:
                    d = plane.distance_point_signed(p.coords)
                    distances.append(float(f"{d:.2e}"))
                f_errors[f.index] = distances
    if f_errors:
        raise ValueError({
            "name": name,
            "error": "nonplanar_faces",
            "error_nonplanar_faces": f_errors
        })

    # find planes...
    planes:PlaneSet = set()
    plane_max_points = 0

    # add faces as planes
    for f in mesh.faces:
        planes.add(frozenset(v.index for v in f.points))

    # find non-face planes
    start_time = time.time()
    point_nums = set(range(len(mesh.points)))
    for combo in (set(c) for c in itertools.combinations(point_nums, 3)):
        if any(combo <= p for p in planes):
            continue # already have this combo

        plane = Plane.from_points(*[mesh.points[i].coords for i in combo])
        plane_points = combo.copy()

        for i in (point_nums - combo):
            if plane.distance_point(mesh.points[i].coords) < TOLERANCE:
                plane_points.add(i)

        planes.add(frozenset(plane_points))
        plane_max_points = max(plane_max_points, len(plane_points))

    t = round(time.time() - start_time, 2)
    print(f"[{name}] faces: {len(mesh.faces)}\t\tbiggest: {max(len(f.points) for f in mesh.faces)}\tplanes: {len(planes)}\tmax points: {plane_max_points}\ttime: {t}")
    return planes


def find_all_planes(db:shelve.Shelf, overwrite=False) -> None:
    """find planes for all meshes in db. If a mesh already has planes, they will be recalculated and overwritten if overwrite=False, otherwise the mesh will be skipped"""
    for name in db:
        if overwrite or ("planes" not in db[name]) or (not db[name]["planes"]):
            try:
                db[name]["planes"] = find_planes(db[name]["mesh"], name)
                db.sync()
            except ValueError as e:
                print(e)


def equal_planes(a:dict[int,PlaneSet], b:dict[int,PlaneSet]) -> bool:
    """check if the given plane set collections are equal"""
    #todo?: show differences?
    return a == b  # lol


def find_upper_bound(mesh:off.Mesh, name:str=None, print_=False) -> tuple[int,str]:
    """find an upper bound for mimimum set cover"""  # this might be more useful if I decided to write my own set-cover algorithm
    v_over_3 = math.ceil(len(mesh.points) / 3)
    unique_x = len(set(v.x for v in mesh.points))
    unique_y = len(set(v.y for v in mesh.points))
    unique_z = len(set(v.z for v in mesh.points))
    values = (v_over_3, unique_x, unique_y, unique_z)
    ub = min(values)

    reason = ""
    if v_over_3 == ub:
        reason += " v/3"
    if unique_x == ub:
        reason += " unique_x"
    if unique_y == ub:
        reason += " unique_y"
    if unique_z == ub:
        reason += " unique_z"

    if print_:
        msg = f"{name} {values} | starting bssf = {ub} | from:"
        print(msg + reason)
    return (ub, reason.strip())


# def find_bins(all_planes:dict[int,PlaneSet]) -> None:
#     """find the number of planes of each size (number of covered points) for each plane set and save it as a .csv"""
#     bins:dict[int,dict[int,int]] = {}
#     for j_number, planes in all_planes.items():
#         bins[j_number] = {}
#         for p in planes:
#             s = len(p)
#             if s not in bins[j_number]:
#                 bins[j_number][s] = 0
#             bins[j_number][s] += 1
#     m = max(max(x.keys()) for x in bins.values())

#     with open("bins.csv", "w") as f:
#         f.write(f"J,{','.join(str(b) for b in range(3,m+1))},total\n")
#         for j in bins:
#             f.write(f"{j},{','.join(str(bins[j][b] if b in bins[j] else 0) for b in range(3,m+1))},{len(all_planes[j])}\n")


def minimum_covering_planes(db:shelve.Shelf, name:str) -> tuple[int,frozenset[frozenset[int]],float]:
    mesh:off.Mesh = db[name]["mesh"]
    planes_list:list[frozenset[int]] = list(db[name]["planes"])
    plane_sort_key = lambda x: (len(x), sorted(x))
    planes_list.sort(key=plane_sort_key, reverse=True)

    matrix = np.zeros((len(planes_list), len(mesh.points)), dtype=np.byte)
    for i in range(len(planes_list)):
        for j in planes_list[i]:
            matrix[i,j] = 1

    sc = setcover.SetCover(matrix.T, cost=np.ones((len(planes_list),), dtype=np.byte))
    solution_size, minutes = sc.SolveSCP()
    solution_size = int(solution_size)

    ub, reason = find_upper_bound(mesh)
    if solution_size > ub:
        print(f"[{name}] found solution {solution_size} is worse than upper bound {ub} ({reason})")

    solution_set_indexes = [int(i) for i in range(len(sc.s)) if sc.s[i]]
    solution = frozenset(planes_list[i] for i in solution_set_indexes)

    if db[name]["best_solution"] is None:
        db[name]["best_solution"] = solution_size
        db[name]["solutions"] = [solution]
        db[name]["num_solutions"] = 1
        print(f"[{name}] Found solution: {solution_size}")
    elif solution_size < db[name]["best_solution"]:
        db[name]["best_solution"] = solution_size
        db[name]["solutions"] = [solution]
        db[name]["num_solutions"] = 1
        print(f"[{name}] Found better solution: {solution_size}")
    elif db[name]["best_solution"] == solution_size and solution not in db[name]["solutions"]:
        db[name]["solutions"].append(solution)
        db[name]["num_solutions"] += 1
        print(f"[{name}]\tFound alternate solution: {solution_size}")

    db.sync()

    return (solution_size, solution, minutes*60)


def all_minimum_covering_planes(db:shelve.Shelf, range_:Sequence[int]=None) -> None:
    for name in db:
        if range_ is None or (range_[0] <= db[name]["best_solution"] <= range_[1]):
            minimum_covering_planes(db, name)


def str_solutions(db:shelve.Shelf, sep="\t") -> str:
    result = sep.join(("name","best_solution","num_solutions","upper_bound","ub_reason")) + "\n"
    for name in db:
        result += sep.join(str(i) for i in (
            name,
            db[name]["best_solution"],
            db[name]["num_solutions"],
            db[name]["upper_bound"],
            db[name]["ub_reason"]
        )) + "\n"
    return result


def print_solutions(db:shelve.Shelf) -> None:
    print(str_solutions(db, "\t"))


def save_solutions_csv(db:shelve.Shelf, outfile:str):
    with open(outfile, "w") as f:
        f.write(str_solutions(db, ","))


def plot_solution(mesh:off.Mesh, planes_fs:frozenset[frozenset[int]], origin_vectors:float=0):
    planes = list(planes_fs)
    covered_points:set[int] = set()
    point_colors:dict[int,int] = {}

    for i, plane in enumerate(planes):
        for point_index in plane:
            if point_index not in covered_points:
                covered_points.add(point_index)
                point_colors[point_index] = i

    ax:Axes3D
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_axis_off()
    colors = matplotlib.color_sequences["petroff10"]

    # origin vectors
    for vec, color in (
        ((origin_vectors,0,0), "red"),
        ((0,origin_vectors,0), "green"),
        ((0,0,origin_vectors), "blue")
    ):
        v = Vector(vec)
        v.plot_3d(ax, color=(color, 0.5))

    # vertices
    for i, point in enumerate(mesh.points):
        p = Point(point.coords)
        p.plot_3d(ax, color=colors[point_colors[i] % len(colors)])

    # edges
    for e in mesh.edges:
        ls = LineSegment(e.p0.coords, e.p1.coords)
        ls.plot_3d(ax, color=("gray", 0.8), lw=1)

    ax.set_aspect("equal")
    xmin, xmax, ymin, ymax, zmin, zmax = ax.get_w_lims()

    # covering planes
    for i, plane in enumerate(planes):
        coords = [mesh.points[j].coords for j in plane]

        pl = Plane.from_points(*coords[:3])
        normal = pl.normal

        xc = sum(c[0] for c in coords) / len(coords)
        yc = sum(c[1] for c in coords) / len(coords)
        zc = sum(c[2] for c in coords) / len(coords)
        center = [xc, yc, zc]

        start = coords[0]
        vector_a = Vector.from_points(center, start)

        angle_coords = []

        for _, c in enumerate(coords):
            vector_b = Vector.from_points(center, c)
            if c == start:
                angle = 0
            elif vector_a.is_parallel(vector_b):
                angle = vector_a.angle_between(vector_b)
            else:
                angle = vector_a.angle_signed_3d(vector_b, normal)
            angle_coords.append((angle, c))

        coords_sorted = [ac[1] for ac in sorted(angle_coords)]

        polygon = art3d.Poly3DCollection([coords_sorted], color=(colors[i % len(colors)], 0.25), lw=0)
        ax.add_collection3d(polygon)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)

    fig.tight_layout()
    plt.show()


def list_database(db:shelve.Shelf) -> None:
    print("name\tvertices\tedges\tfaces\tbest_solution\tnum_solutions\tupper_bound\tub_reason\tnum_planes")
    for name in db:
        mesh:off.Mesh = db[name]["mesh"]
        v = len(mesh.points)
        e = len(mesh.edges)
        f = len(mesh.faces)
        print(f'{name}\t{v}\t{e}\t{f}\t{db[name]["best_solution"]}\t{db[name]["num_solutions"]}\t{db[name]["upper_bound"]}\t{db[name]["ub_reason"]}\t{len(db[name]["planes"])}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title="commands", required=True)

    load_mesh_parser = subparsers.add_parser("load-mesh", aliases=["lm"], help="load a mesh file")
    load_mesh_parser.set_defaults(cmd="load-mesh")
    load_mesh_parser.add_argument("meshfile", help=".off file to load")
    load_mesh_parser.add_argument("--clear", action="store_true", help="if the mesh already exists in the db, delete the entire db entry (planes, solutions, etc) for the mesh before reloading it")

    load_meshes_parser = subparsers.add_parser("load-mesh-dir", aliases=["lmd"], help="load all mesh files in a directory")
    load_meshes_parser.set_defaults(cmd="load-mesh-dir")
    load_meshes_parser.add_argument("dir", help="directory to search for .off files to load")
    load_meshes_parser.add_argument("--clear", action="store_true", help="if a mesh already exists in the db, delete the entire db entry (planes, solutions, etc) for the mesh before reloading it")

    find_planes_parser = subparsers.add_parser("find-planes", aliases=["fp"], help="find all sets of coplanar vertices for loaded meshes")
    find_planes_parser.set_defaults(cmd="find-planes")
    find_planes_parser.add_argument("--overwrite", action="store_true", help="recalculate and overwrite planes (otherwise meshes with planes already will be skipped)")

    minimum_covering_planes_parser = subparsers.add_parser("minimum-covering-planes", aliases=["mcp"], help="find the minimum covering planes of the given mesh")
    minimum_covering_planes_parser.set_defaults(cmd="minimum-covering-planes")
    minimum_covering_planes_parser.add_argument("name", help="name of the mesh")
    minimum_covering_planes_parser.add_argument("--quiet", "-q", action="store_true", help="don't print full solution. Only print when there's a new solution")

    all_minimum_covering_planes_parser = subparsers.add_parser("all-minimum-covering-planes", aliases=["amcp"], help="find the minimum covering planes of all meshes")
    all_minimum_covering_planes_parser.set_defaults(cmd="all-minimum-covering-planes")
    all_minimum_covering_planes_parser.add_argument("--solution-range", "-r", nargs=2, type=int, metavar=("MIN", "MAX"), help="only recalculate solutions in the given range")

    show_solutions_parser = subparsers.add_parser("show-solutions", aliases=["show", "solutions", "s"], help="show or save the calculated solutions")
    show_solutions_parser.set_defaults(cmd="show-solutions")
    show_solutions_parser.add_argument("--output", "-o", help=".csv file to save to instead of printing")

    list_parser = subparsers.add_parser("list", aliases=["l"], help="list some stuff about meshes in the database")
    list_parser.set_defaults(cmd="list")

    plot_solution_parser = subparsers.add_parser("plot-solution", aliases=["plot", "p"], help="make a 3d plot of a solution")
    plot_solution_parser.set_defaults(cmd="plot-solution")
    plot_solution_parser.add_argument("name", help="name of the mesh")
    plot_solution_parser.add_argument("--solution", "-s", type=int, default=0, help="solution number")
    plot_solution_parser.add_argument("--origin-vectors", "-O", type=float, default=0, metavar="SIZE", help="show x, y, and z vectors from the origin of the given size")

    args = parser.parse_args()

    with open_db() as db:
        if args.cmd == "load-mesh":
            load_mesh(db, args.meshfile, args.clear)
        elif args.cmd == "load-mesh-dir":
            load_meshes(db, args.dir, args.clear)
        elif args.cmd == "find-planes":
            find_all_planes(db, args.overwrite)
        elif args.cmd == "minimum-covering-planes":
            solution_size, solution, t_seconds = minimum_covering_planes(db, args.name)
            if not args.quiet:
                print(f"[{args.name}] found solution of {solution_size} in {t_seconds:.2f} seconds: {[list(i) for i in solution]}")
        elif args.cmd == "all-minimum-covering-planes":
            all_minimum_covering_planes(db, args.solution_range)
        elif args.cmd == "show-solutions":
            if args.output is not None:
                save_solutions_csv(db, args.output)
            else:
                print_solutions(db)
        elif args.cmd == "list":
            list_database(db)
        elif args.cmd == "plot-solution":
            mesh = db[args.name]["mesh"]
            solution = db[args.name]["solutions"][args.solution]
            plot_solution(mesh, solution, args.origin_vectors)
