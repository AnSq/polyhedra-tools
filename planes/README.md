# Minimum Covering Planes

## What is this?

This program finds the "minimum vertex-covering planes" of a polyhedron.
"Vertex-covering planes" are a set of planes that together contain every vertex
of a polyhedron. The "minimum vertex-covering planes" (or just "minimum covering
planes" or "MCP"), therefore, is the smallest such set of planes. A polyhedron
my have one MCP set, or many.

<!--## Installation

TODO-->

## Usage

`planes.py` uses a persistent database to store information about meshes and
solutions, and different commands to run different steps of the process.

First, load the mesh or meshes. `planes` supports 3D
[OFF meshes](https://en.wikipedia.org/wiki/OFF_(file_format)) with coplanar
faces. Mesh filenames should be of the form `<name>_<fullname>.off`, where
`<name>` is a short name used as to refer to the mesh is some commands, and
`<fullname>` is the full name of the polyhedron. For example, `C_Cube.off` or
`J1_Square_pyramid.off`.

    planes.py load-mesh <meshfile>
    # or #
    planes.py load-mesh-dir <directory>

Then find *all* possible planes that might be included in a solution. This is a
$O(n^2)$ operation, and may take several minutes for large collections of meshes
or meshes with many vertices.

    planes.py find-planes

Finally, find the MCP for each mesh:

    planes.py all-minimum-covering-planes

You will now have a database that contains *a* MCP solution for every mesh you
loaded. To see them, use the `list` command. The "Best Solution" column shows
the minimum MCP size currently calculated.

    planes.py list

To see the planes themselves, use the `plot` command. This will open an
interactive window showing the mesh with the covering planes.

    planes.py plot <mesh_name>

At this point though, you will probably *not* have the actual minimum solution
for some or all of your meshes. This is because finding the MCP is an instance
of the [set cover problem](https://en.wikipedia.org/wiki/Set_cover_problem),
which is known to be [NP-hard](https://en.wikipedia.org/wiki/NP-hardness),
meaning the only 'efficient' algorithms for solving it are approximate ones.
This program makes use of a library that uses randomization in its set cover
algorithm, so running it multiple times may result in a better solution.

You can re-run the `minimum-covering-planes` or `all-minimum-covering-planes`
commands, optionally with the `--loop` flag, to keep trying for better
solutions. See their in-program `--help` pages for more details. It may take
several hours of random trying to find the best results.

## Implementation Details

### Upper Bound

The program uses a couple of simple heuristics to determine an upper bound for
the best MCP solution. The first is calculating $\lceil v/3 \rceil$, where $v$
is the number of vertices of the mesh. Since any three points are coplanar,
there's always a solution consisting only of planes that cover three points.

The second heuristic is to find the number of unique $x$, $y$, or $z$
coordinates of vertices. These correspond to solutions consisting only of planes
that are parallel to the same coordinate plane. This depends on mesh
orientation, so rotating the mesh before importing it may give a better upper
bound.

### Lower Bound

The program also calculates a crude lower bound for the best solution. This is
$\lceil v/l \rceil$, where $l$ is the maximum number of vertices covered by any
plane (the "largest" plane). If a solution exists that consists only of such
"largest planes", then the actual best solution will equal the lower bound.
There is *no guarantee* that this is true though, and the best possible solution
will often be higher than the "lower bound".

<!--### Assumptions

TODO

* coplanar faces
* convex-->

<!--### Limitations

TODO-->

### The Database

The `planes.py` database uses the Python
[`shelve`](https://docs.python.org/3.13/library/shelve.html) module. In Python
3.13+, this is backed by a SQLite database. In earlier versions, it uses one of
the other [`dbm` backends](https://docs.python.org/3.13/library/dbm.html).
I have only tested it with Python 3.13.

The Python documentation gives [the following warning](https://docs.python.org/3.13/library/shelve.html#shelve-security)
about `shelve` databases that applies to all versions:

> **Warning:** Because the `shelve` module is backed by
> [`pickle`](https://docs.python.org/3.13/library/pickle.html), it is insecure
> to load a shelf from an untrusted source. Like with pickle, loading a shelf
can execute arbitrary code.

[See below](#database-import--export) for information on how to import and
export the database as a JSON file instead.

## Advanced Usage

### Using Multiple Databases

By default, `planes.py` uses a database called `database.db`. You can use a
different database using the `--database` option *before* the command. For
example:

    planes.py --database second_database.db load-mesh <meshfile>

This works with all commands.

The actual database file may have additional suffixes appended to it, and there
may be multiple files per database, depending on your Python version and
platform.

### Database Import & Export

To avoid the security issues with sharing `shelve` databases ([see above](#the-database)),
`planes.py` supports importing and exporting the database in JSON format. JSON
is a much safer format, and loading it does not have the potential to execute
arbitrary code. You still of course have to trust *this* program, but that's
much easier to inspect than a `pickle` object.

To export the database, use:

    planes.py export-database <json_file>

To import a JSON file as a new database, use:

    planes.py --database <database_file> import-database <json_file>

The database must be empty or the import will be rejected.

<!--### The `list` command

TODO-->

<!--### Filters

TODO-->

<!--### Advanced Plotting and Animation

TODO--

<!--## Why does this exist?

TODO-->
