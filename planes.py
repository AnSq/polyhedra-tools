#!/usr/bin/env python
import os
import re
import time
import itertools
import math
import shelve
import logging
import argparse
import functools
import csv

from typing import Callable
from collections.abc import Sequence

from skspatial.objects import Plane, Point, Points, LineSegment, Vector

import numpy as np
from SetCoverPy import setcover  #type: ignore

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D  #type: ignore
import mpl_toolkits.mplot3d.art3d as art3d  #type: ignore
import matplotlib.animation as animation  #type: ignore
from matplotlib.figure import Figure
import matplotlib.patches
import matplotlib.lines
import matplotlib.legend_handler

from tabulate import tabulate
from tqdm import tqdm  #type: ignore

import off

PlaneSet = set[frozenset[int]]

logging.captureWarnings(True)
file_handler = logging.FileHandler(filename="planes.log", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)  # file gets everything from this module
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.WARNING)  # stderr only gets warning and up
logging.basicConfig(handlers=[file_handler, stream_handler], level=logging.WARNING)  # the default for imported modules is warning and up
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # this module logs everything, subject to the notes above on the handlers
setcover_logger = logging.getLogger("setcover")

setcover.np.seterr(divide="ignore", invalid="ignore")

TOLERANCE = 1e-12
DB_FILE = "database.db"



class Filter:
    """controls which meshes are processed in an operation"""

    LOWER_BOUND_OPTIONS = ("above", "at", "below")

    def __init__(self, name_re:str=None, solution_range:Sequence[int]=None, lower_bound:str=None) -> None:

        if lower_bound not in self.LOWER_BOUND_OPTIONS + (None,):
            raise ValueError(f"lower_bound must be one of {self.LOWER_BOUND_OPTIONS + (None,)}")

        self.name_re = re.compile(name_re) if (name_re is not None) else None
        self.solution_range = solution_range
        self.lower_bound = lower_bound

    def __repr__(self) -> str:
        return f"Filter({repr(None if not self.name_re else self.name_re.pattern)}, {self.solution_range}, {self.lower_bound})"

    def __call__(self, db:shelve.Shelf) -> list[str]:
        """return a list of db keys (names) that pass the filter"""
        result = []
        for name in db:
            best_solution = db[name]["best_solution"] or 0
            lb = find_lower_bound(db[name]["mesh"], db[name]["planes"])

            if ((self.name_re is None or self.name_re.search(name))
            and (self.solution_range is None or (self.solution_range[0] <= best_solution <= self.solution_range[1]))
            and (
                self.lower_bound is None
                or (self.lower_bound == "above" and best_solution > lb)
                or (self.lower_bound == "at" and best_solution == lb)
                or (self.lower_bound == "below" and best_solution < lb)
            )):
                result.append(name)

        return result



def open_db(fname=DB_FILE) -> shelve.Shelf:
    return shelve.open(fname, writeback=True)


def load_mesh(db:shelve.Shelf, fname:str, clear=False) -> tuple[str,off.Mesh] | tuple[None,None]:
    """load the given mesh file into the db. Deletes the entire db entry (planes, solutions, etc) for the mesh first if clear=True"""
    name, fullname = os.path.basename(fname).split(".")[0].split("_", 1)
    fullname = fullname.replace("_", " ")

    if clear:
        if name in db:
            row = db[name]
            print(f"\nThe mesh {name} is already in the database:")
            print(f"Full name: {row['fullname']}")
            print(f"Mesh: {row['mesh']}")
            print(f"Upper bound: {row['upper_bound']} ({row['ub_reason']})")
            print(f"Best solution: {row['best_solution']} ({row['num_solutions']} solutions)")
            print(f"Total planes: {len(row['planes'])}")
            confirm = input(f"Are you sure you want to delete this entire database entry and replace it with the new mesh from '{fname}'?: ")
            if confirm == "yes":
                del db[name]
            else:
                print('Your answer was not "yes", so the mesh will not be loaded and the database will not be changed.')
                return (None, None)

    mesh = off.Mesh.load(fname, info=True)

    if name not in db:
        ub, reason = find_upper_bound(mesh)
        db[name] = {
            "name"          : name,
            "fullname"      : fullname,
            "upper_bound"   : ub,
            "ub_reason"     : reason,
            "best_solution" : None,
            "num_solutions" : 0,
            "solutions"     : [],
            "planes"        : [],
            "lower_bound"   : 0,
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
    point_nums = set(range(mesh.num_points))
    for combo in tqdm((set(c) for c in itertools.combinations(point_nums, 3)), desc=f"Finding planes for {name}", leave=False, position=1, total=math.comb(len(point_nums), 3)):
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
    logger.info(f"[{name}] faces: {mesh.num_faces}\t\tbiggest: {max(len(f.points) for f in mesh.faces)}\tplanes: {len(planes)}\tmax points: {plane_max_points}\ttime: {t}")
    return planes


def find_all_planes(db:shelve.Shelf, db_filter:Filter, overwrite=False) -> int:
    """find planes for all meshes in db. If a mesh already has planes, they will be recalculated and overwritten if overwrite=True, otherwise the mesh will be skipped.
    Returns the number of meshes that were processed"""
    filter_matches = db_filter(db)
    names = []
    for name in filter_matches:
        if overwrite or ("planes" not in db[name]) or (not db[name]["planes"]):
            names.append(name)

    for name in (progress_bar := tqdm(names, desc="Finding all planes")):
        progress_bar.set_postfix({"current": name})
        try:
            db[name]["planes"] = find_planes(db[name]["mesh"], name)
            db[name]["lower_bound"] = find_lower_bound(db[name]["mesh"], db[name]["planes"])
            db.sync()
        except ValueError as e:
            logger.error(e)

    return len(names)


def find_upper_bound(mesh:off.Mesh, name:str=None, print_=False) -> tuple[int,str]:
    """find an upper bound for mimimum set cover"""  # this might be more useful if I decided to write my own set-cover algorithm
    v_over_3 = math.ceil(mesh.num_points / 3)
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
        msg = f"{name} {values} | upper bound = {ub} | from:"
        print(msg + reason)
    return (ub, reason.strip())


def find_lower_bound(mesh:off.Mesh, planes:PlaneSet) -> int:
    largest_plane = max(len(p) for p in planes)
    return math.ceil(mesh.num_points / largest_plane)


def _add_lower_bounds(db:shelve.Shelf) -> None:
    """add lower_bound to a database that didn't originally have it"""
    for name in db:
        db[name]["lower_bound"] = find_lower_bound(db[name]["mesh"], db[name]["planes"])
        db.sync()


def minimum_covering_planes(db:shelve.Shelf, name:str, exclude_3s=False) -> tuple[int,frozenset[frozenset[int]],float]:
    # logger.debug(f"calculating minimum covering planes of {name}")
    mesh:off.Mesh = db[name]["mesh"]
    planes_list:list[frozenset[int]] = list(db[name]["planes"])
    plane_sort_key = lambda x: (len(x), sorted(x))
    planes_list.sort(key=plane_sort_key, reverse=True)

    if exclude_3s:
        planes_list = [p for p in planes_list if len(p) > 3]

    matrix = np.zeros((len(planes_list), mesh.num_points), dtype=np.byte)
    for i in range(len(planes_list)):
        for j in planes_list[i]:
            matrix[i,j] = 1

    sc = setcover.SetCover(matrix.T, cost=np.ones((len(planes_list),), dtype=np.byte), maxiters=1)
    solution_size, minutes = sc.SolveSCP()
    seconds = minutes * 60
    solution_size = int(solution_size)

    oversize_warning = ""
    if solution_size > db[name]["upper_bound"]:
        oversize_warning = f' (worse than known upper bound {db[name]["upper_bound"]})'

    solution_set_indexes = [int(i) for i in range(len(sc.s)) if sc.s[i]]
    solution = frozenset(planes_list[i] for i in solution_set_indexes)

    if db[name]["best_solution"] is None:
        db[name]["best_solution"] = solution_size
        db[name]["solutions"] = [solution]
        db[name]["num_solutions"] = 1
        logger.info(f"[{name}] Found solution: {solution_size} (in {seconds:.2f}s)" + oversize_warning)
    elif solution_size < db[name]["best_solution"]:
        db[name]["best_solution"] = solution_size
        db[name]["solutions"] = [solution]
        db[name]["num_solutions"] = 1
        logger.info(f"[{name}] Found better solution: {solution_size} (in {seconds:.2f}s)" + oversize_warning)
    elif db[name]["best_solution"] == solution_size and solution not in db[name]["solutions"]:
        db[name]["solutions"].append(solution)
        db[name]["num_solutions"] += 1
        logger.info(f"[{name}]\tFound alternate solution: {solution_size} (in {seconds:.2f}s)" + oversize_warning)

    db.sync()

    return (solution_size, solution, seconds)


def all_minimum_covering_planes(db:shelve.Shelf, db_filter:Filter) -> int:
    """calculate minimum covering planes for all meshes that pass the filter. Return how many were processed"""
    matches = db_filter(db)
    for name in (progress_bar := tqdm(matches, desc="Finding all minimum covering planes", leave=False)):
        progress_bar.set_postfix({"current": name})
        minimum_covering_planes(db, name)
    return len(matches)


def find_duplicate_points(planes:list[frozenset[int]]) -> list[bool]:
    result = []
    for i, plane in enumerate(planes):
        result.append(
            # any point in this plane is in any other plane
            any(any(point in other_plane for other_plane in planes[:i] + planes[i+1:]) for point in plane)
        )
    return result


def plot_solution(mesh:off.Mesh, planes_fs:frozenset[frozenset[int]], title:str, origin_vectors:float=0, show_edges=True, label_points=False):
    ax:Axes3D
    fig, ax = plt.subplots(figsize=(4.8, 4.8), subplot_kw={"projection": "3d"})
    ax.set_axis_off()
    colors = matplotlib.color_sequences["petroff10"]
    fontfamily = "Montserrat"
    matplotlib.rc("font", family=fontfamily)

    planes = list(planes_fs)
    planes.sort(key=lambda x: (-len(x), sorted(list(x))))

    plane_has_duplicate_points = find_duplicate_points(planes)

    # determine point colors
    point_colors:dict[int,int] = {}
    if planes:
        covered_points:set[int] = set()
        for i, plane in enumerate(planes):
            for point_index in plane:
                if point_index not in covered_points:
                    covered_points.add(point_index)
                    point_colors[point_index] = i
    else:
        for point in mesh.points:
            point_colors[point.index] = 0

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
        if label_points:
            ax.text(point.x, point.y, point.z, i)

    # edges
    if show_edges:
        for e in mesh.edges:
            ls = LineSegment(e.p0.coords, e.p1.coords)
            ls.plot_3d(ax, color=("gray", 0.5), linewidth=0.5)

    ax.set_aspect("equal")
    xmin, xmax, ymin, ymax, zmin, zmax = ax.get_w_lims()

    # covering planes
    legend_patches = []
    legend_labels = []
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
                angle:float = 0
            elif vector_a.is_parallel(vector_b):
                angle = vector_a.angle_between(vector_b)
            else:
                angle = vector_a.angle_signed_3d(vector_b, normal)
            angle_coords.append((angle, c))

        coords_sorted = [ac[1] for ac in sorted(angle_coords)]

        basecolor = colors[i % len(colors)]
        transcolor = (basecolor, 0.25)
        polygon = art3d.Poly3DCollection([coords_sorted], color=transcolor, linewidth=0)
        legend_patches.append((
            matplotlib.lines.Line2D([], [], color=basecolor, marker=".", markersize=9, linewidth=0),
            matplotlib.patches.Patch(color=transcolor, linewidth=0)  #type: ignore
        ))
        legend_labels.append(f"{len(coords)} vertices{'*' if plane_has_duplicate_points[i] else ''}")
        ax.add_collection3d(polygon)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)

    fig.tight_layout()
    pad = 0.1
    fig.subplots_adjust(left=-pad, bottom=-pad, right=1+pad, top=1)

    ax.set_title(title, y=1+pad, pad=-28, fontfamily=[fontfamily], fontweight="medium")

    # mesh info legend
    meshinfo = fig.legend(handles=(
        matplotlib.lines.Line2D([], [], lw=0, ms=0, label=f"{mesh.num_points} vertices"),
        matplotlib.lines.Line2D([], [], lw=0, ms=0, label=f"{mesh.num_edges} edges"),
        matplotlib.lines.Line2D([], [], lw=0, ms=0, label=f"{mesh.num_faces} faces")
    ), loc="lower left", fontsize="small", title="Mesh", title_fontproperties={"weight":"medium"}, handlelength=0)
    fig.add_artist(meshinfo)

    # solution legend
    fig.legend(
        handles=legend_patches,
        labels=legend_labels,
        loc="lower right",
        title=f"{len(planes)} planes",
        title_fontproperties={"weight":"medium"},
        fontsize="small",
        handler_map={tuple: matplotlib.legend_handler.HandlerTuple(None, 0.2)}
    )

    return (fig, ax)


def rotation_function(x:int, total:int) -> tuple[float,float,float]:
    """takes a frame number and the total number of frames and returns (elev, azim, roll) to apply to view"""
    elev = 30 * math.sin(math.radians(360 * x / total))
    azim = 360 * x / total % 360
    roll = 0
    return (elev, azim, roll)


def animation_rotate(i:int, *, ax:Axes3D, total:int):
    ax.view_init(*rotation_function(i, total))


def animate_solution(fig:Figure, ax:Axes3D, anim_func:Callable, num_frames:int, fname:str, tqdm_position:int=0):
    anim = animation.FuncAnimation(fig, func=functools.partial(anim_func, ax=ax, total=num_frames), frames=num_frames, interval=33)
    writer = animation.FFMpegWriter(fps=30, codec="libvpx-vp9", extra_args=["-crf", "30", "-b:v", "0"])
    os.makedirs("output/animation", exist_ok=True)
    with tqdm(total=num_frames, desc=f"Saving animation {fname}", position=tqdm_position) as progress_bar:
        anim.save(f"output/animation/{fname}.webm", writer, progress_callback=lambda c,t: progress_bar.update(1))


TABLE_COLUMN_DEFS = {
    "n": ("Name",           lambda r: r["name"]),
    "N": ("Full Name",      lambda r: r["fullname"]),
    "v": ("Vertices",       lambda r: r["mesh"].num_points),
    "e": ("Edges",          lambda r: r["mesh"].num_edges),
    "f": ("Faces",          lambda r: r["mesh"].num_faces),
    "s": ("Best Solution",  lambda r: r["best_solution"]),
    "S": ("Num. Solutions", lambda r: r["num_solutions"]),
    "u": ("Upper Bound",    lambda r: r["upper_bound"]),
    "U": ("U.B. Reason",    lambda r: r["ub_reason"]),
    "p": ("Total Planes",   lambda r: len(r["planes"])),
    "l": ("Lower Bound",    lambda r: r["lower_bound"])
}
ALL_COLUMNS = "nvefsSuUlpN"

def list_database(db:shelve.Shelf, db_filter:Filter=Filter(), columns=ALL_COLUMNS, csv_file:str=None) -> None:
    if columns == "a":
        columns = ALL_COLUMNS

    headers = []
    for c in columns:
            headers.append(TABLE_COLUMN_DEFS[c][0])

    table = []
    for name in db_filter(db):
        table_row = []
        for c in columns:
            table_row.append(TABLE_COLUMN_DEFS[c][1](db[name]))
        table.append(table_row)

    if csv_file:
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(table)
    else:
        print(tabulate(table, headers=headers, tablefmt="rounded_outline"))
        print(f"{len(table)} records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title="commands", required=True)

    loopable_parser = argparse.ArgumentParser(add_help=False)
    loopable_parser.add_argument("--loop", "-l", action="store_true", help="loop forever (ctrl-C to exit)")

    filterable_parser = argparse.ArgumentParser(add_help=False)
    filterable_parser.set_defaults(filterable=True)
    filterable_parser_group = filterable_parser.add_argument_group("Filter options")
    filterable_parser_group.add_argument("--solution-range", "-r", nargs=2, type=int, metavar=("MIN", "MAX"), help="only process meshes with solutions in the given range")
    filterable_parser_group.add_argument("--name-re", "-n", help="only process meshes with names matching the given regex")
    filterable_parser_group.add_argument("--lower-bound", "--lb", choices=Filter.LOWER_BOUND_OPTIONS, help="filter by solution size's relationship to the lower bound")

    load_mesh_parser = subparsers.add_parser("load-mesh", aliases=["lm"], help="load a mesh file")
    load_mesh_parser.set_defaults(cmd="load-mesh")
    load_mesh_parser.add_argument("meshfile", help=".off file to load")
    load_mesh_parser.add_argument("--clear", action="store_true", help="if the mesh already exists in the db, delete the entire db entry (planes, solutions, etc) for the mesh before reloading it")

    load_meshes_parser = subparsers.add_parser("load-mesh-dir", aliases=["lmd"], help="load all mesh files in a directory")
    load_meshes_parser.set_defaults(cmd="load-mesh-dir")
    load_meshes_parser.add_argument("dir", help="directory to search for .off files to load")
    load_meshes_parser.add_argument("--clear", action="store_true", help="if a mesh already exists in the db, delete the entire db entry (planes, solutions, etc) for the mesh before reloading it")

    find_planes_parser = subparsers.add_parser("find-planes", aliases=["fp"], help="find all sets of coplanar vertices for loaded meshes", parents=[filterable_parser])
    find_planes_parser.set_defaults(cmd="find-planes")
    find_planes_parser.add_argument("--overwrite", action="store_true", help="recalculate and overwrite planes (otherwise meshes with planes already will be skipped)")

    minimum_covering_planes_parser = subparsers.add_parser("minimum-covering-planes", aliases=["mcp"], help="find the minimum covering planes of the given mesh", parents=[loopable_parser])
    minimum_covering_planes_parser.set_defaults(cmd="minimum-covering-planes")
    minimum_covering_planes_parser.add_argument("name", help="name of the mesh")
    minimum_covering_planes_parser.add_argument("--quiet", "-q", action="store_true", help="don't print full solution. Only print when there's a new solution")
    minimum_covering_planes_parser.add_argument("--exclude-3s", "-3", action="store_true", help="don't consider 3-point planes when finding a solution")

    all_minimum_covering_planes_parser = subparsers.add_parser("all-minimum-covering-planes", aliases=["amcp"], help="find the minimum covering planes of all meshes", parents=[loopable_parser, filterable_parser])
    all_minimum_covering_planes_parser.set_defaults(cmd="all-minimum-covering-planes")

    show_solutions_parser = subparsers.add_parser("show-solutions", aliases=["show", "solutions", "s"], help="show or save the calculated solutions", parents=[filterable_parser])
    show_solutions_parser.set_defaults(cmd="show-solutions")
    show_solutions_parser.add_argument("--output", "-o", help=".csv file to save to instead of printing")

    list_parser = subparsers.add_parser(
        "list",
        aliases=["l"],
        help="list some stuff about meshes in the database",
        epilog=f"Available columns are:\n{chr(10).join(f'  {k}: {v[0]}' for k,v in TABLE_COLUMN_DEFS.items())}\nYou can also use the special column option 'a', which is equivalent to '{ALL_COLUMNS}'",  # chr(10) is newline, because backslashes "aren't allowed" in f-string expressions
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[filterable_parser]
    )
    list_parser.set_defaults(cmd="list")
    list_parser.add_argument("--columns", "-c", default="nsSuU", help="which columns to show. See below for options. Default: '%(default)s'")
    list_parser.add_argument("--csv", metavar="FNAME", dest="csv_file", help="write to the given CSV file instead of printing a table")

    # vvv   Plotting Args   vvv

    plot_parser = argparse.ArgumentParser(add_help=False)
    plot_parser.add_argument("--origin-vectors", "-O", type=float, default=0, metavar="SIZE", help="show x, y, and z vectors from the origin of the given size")
    plot_parser.add_argument("--hide-edges", "-e", action="store_true", help="don't show mesh edges in plot")
    plot_parser.add_argument("--label-points", "-p", action="store_true", help="show point index numbers")

    single_plot_parser = argparse.ArgumentParser(add_help=False)
    single_plot_parser.add_argument("name", help="which mesh to show")

    solution_plot_parser = argparse.ArgumentParser(add_help=False)
    solution_plot_parser.add_argument("--solution", "-s", type=int, default=0, help="solution number")

    plot_mesh_parser = subparsers.add_parser("plot-mesh", aliases=["pm"], help="make a 3d plot of a mesh, without showing solution", parents=[plot_parser, single_plot_parser])
    plot_mesh_parser.set_defaults(cmd="plot-mesh")

    plot_solution_parser = subparsers.add_parser("plot-solution", aliases=["plot", "p"], help="make a 3d plot of a solution", parents=[plot_parser, single_plot_parser, solution_plot_parser])
    plot_solution_parser.set_defaults(cmd="plot-solution")

    animate_solution_parser = subparsers.add_parser("animate-solution", aliases=["a"], help="animate a solution rotating", parents=[plot_parser, single_plot_parser, solution_plot_parser])
    animate_solution_parser.set_defaults(cmd="animate-solution", label_points=False)

    save_plots_parser = subparsers.add_parser("save-plots", aliases=["sp"], help="save plot images", parents=[plot_parser, filterable_parser])
    save_plots_parser.set_defaults(cmd="save-plots")

    save_all_plots_parser = subparsers.add_parser("save-all-plots", aliases=["sap"], help="save plot images for every solution in the database", parents=[plot_parser, filterable_parser])
    save_all_plots_parser.set_defaults(cmd="save-all-plots")

    # ^^^   Plotting Args   ^^^

    test_filter_parser = subparsers.add_parser("test-filter", aliases=["tf"], help="list which meshes pass the given filter", parents=[filterable_parser])
    test_filter_parser.set_defaults(cmd="test-filter")

    args = parser.parse_args()
    if getattr(args, "filterable", False):
        db_filter = Filter(args.name_re, args.solution_range, args.lower_bound)

    with open_db() as db:
        try:
            if args.cmd == "load-mesh":
                load_mesh(db, args.meshfile, args.clear)

            elif args.cmd == "load-mesh-dir":
                load_meshes(db, args.dir, args.clear)

            elif args.cmd == "find-planes":
                find_all_planes(db, db_filter, args.overwrite)

            elif args.cmd == "minimum-covering-planes":
                if args.loop:
                    stream_handler.setLevel(logging.ERROR)
                try:
                    n_loops = 0
                    with tqdm(desc="Loops", unit="") as status_bar:
                        while True:
                            solution_size, solution, t_seconds = minimum_covering_planes(db, args.name, args.exclude_3s)
                            if not args.quiet:
                                print(f"[{args.name}] found solution of {solution_size} in {t_seconds:.2f} seconds: {[list(i) for i in solution]}")
                            n_loops += 1
                            status_bar.update(1)
                            if not args.loop:
                                break
                            file_handler.flush()
                except KeyboardInterrupt:

                    print(f"\nexiting after {n_loops} loops")

            elif args.cmd == "all-minimum-covering-planes":
                if args.loop:
                    stream_handler.setLevel(logging.ERROR)
                try:
                    n_loops = 0
                    with tqdm(desc="Loops", unit="", position=1) as status_bar:
                        while True:
                            num_processed = all_minimum_covering_planes(db, db_filter)
                            if not num_processed:
                                break
                            n_loops += 1
                            status_bar.update(1)
                            if not args.loop:
                                break
                            file_handler.flush()
                except KeyboardInterrupt:
                    print(f"\nexiting after {n_loops} loops")

            elif args.cmd == "list":
                list_database(db, db_filter, args.columns, args.csv_file)

            elif args.cmd in ("plot-mesh", "plot-solution", "animate-solution"):
                row = db[args.name]
                mesh = row["mesh"]
                if args.cmd == "plot-mesh":
                    solution = frozenset()
                else:
                    solution = row["solutions"][args.solution]
                title = f'{row["fullname"]} ({args.name})'
                fig, ax = plot_solution(mesh, solution, title, args.origin_vectors, not args.hide_edges, args.label_points)
                if args.cmd in ("plot-mesh", "plot-solution"):
                    plt.show()
                elif args.cmd == "animate-solution":
                    animate_solution(fig, ax, animation_rotate, 120, args.name)

            elif args.cmd == "save-plots":
                for name in (progress_bar := tqdm(db_filter(db), "Saving plots")):
                    progress_bar.set_postfix({"current": name})
                    row = db[name]
                    mesh = row["mesh"]
                    solution = row["solutions"][0]
                    title = f'{row["fullname"]} ({name})'
                    fig, ax = plot_solution(mesh, solution, title, args.origin_vectors, not args.hide_edges, args.label_points)
                    os.makedirs("output/plots", exist_ok=True)
                    fig.savefig(f"output/plots/{name}.png")
                    plt.close(fig)

            elif args.cmd == "save-all-plots":
                for name in (progress_bar := tqdm(db_filter(db), "Saving all plots", position=0)):
                    progress_bar.set_postfix({"current": name})
                    row = db[name]
                    mesh = row["mesh"]
                    title = f'{row["fullname"]} ({name})'
                    os.makedirs(f"output/plots/{name}", exist_ok=True)
                    for i, solution in enumerate(tqdm(row["solutions"], f"Saving {name}", position=1, leave=False)):
                        fig, ax = plot_solution(mesh, solution, title, args.origin_vectors, not args.hide_edges, args.label_points)
                        fig.savefig(f"output/plots/{name}/{name}_{i}.png")
                        plt.close(fig)

            elif args.cmd == "test-filter":
                print(db_filter)
                matches = db_filter(db)
                for name in matches:
                    print(f'{name}\t\t({db[name]["best_solution"]})')
                print(f"({len(matches)} matches)")

        except KeyboardInterrupt:
            print("KeyboardInterrupt")

        db.sync()
        db.close()

    for h in logger.handlers:
        h.flush()
        h.close()


#TODO: filter by above upper bound
