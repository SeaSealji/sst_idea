"""Independent CPU geometry reference inspired by CGFM Algorithm 1.

No upstream code imported. Angles in degrees, distances on the unit sphere.
Legal wet edges must be supplied; graph coordinates alone never create edges.
"""
from dataclasses import dataclass
import math


@dataclass
class SphereGraph:
    latlon: list  # [(latitude_degrees, longitude_degrees), ...]
    area: list
    edges: object  # iterable of undirected (node_a, node_b), any order


def angular_distance(a, b):
    """Great-circle angle, robust near zero and the antipode; longitude wraps."""
    def xyz(ll):
        lat, lon = map(math.radians, ll)
        return (math.cos(lat)*math.cos(lon), math.cos(lat)*math.sin(lon), math.sin(lat))
    u, v = xyz(a), xyz(b)
    cross = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
    return math.atan2(math.sqrt(sum(t*t for t in cross)), sum(x*y for x, y in zip(u, v)))


def canonical_graph(graph):
    n = len(graph.area)
    if not n or len(graph.latlon) != n:
        raise ValueError('nonempty positions/areas of equal length required')
    area = [float(a) for a in graph.area]
    if any(not math.isfinite(a) or a <= 0 for a in area):
        raise ValueError('positive finite areas required')
    pos = []
    for ll in graph.latlon:
        if len(ll) != 2 or not all(math.isfinite(float(v)) for v in ll):
            raise ValueError('finite latitude/longitude pairs required')
        lat, lon = map(float, ll)
        if not -90 <= lat <= 90:
            raise ValueError('latitude outside [-90,90]')
        pos.append((lat, lon % 360))
    edges = set()
    for pair in graph.edges:
        if len(pair) != 2 or any(type(v) is not int or not 0 <= v < n for v in pair):
            raise ValueError('edges must contain valid integer node pairs')
        a, b = pair
        if a == b:
            raise ValueError('self edges are not legal wet faces')
        edges.add((min(a, b), max(a, b)))
    return SphereGraph(pos, area, edges)


def coarsen(graph, ranking='geometry'):
    """Maximal independent representatives, connected stars, quotient edges.

    geometry: sum of neighbor great-circle angles = degree * mean distance.
    degree: increasing degree; index: increasing node index (Guillard-like).
    Ties use node index, so node relabeling equivariance is NOT promised.
    Undirected adjacency guarantees every dropped node has a retained neighbor.
    Parent is nearest retained neighbor, then index. No iterative propagation.
    """
    if ranking not in ('geometry', 'degree', 'index'):
        raise ValueError('unknown ranking')
    g = canonical_graph(graph)
    n = len(g.area)
    adj = [set() for _ in range(n)]
    distances = {}
    for a, b in sorted(g.edges):
        adj[a].add(b)
        adj[b].add(a)
        distances[a, b] = angular_distance(g.latlon[a], g.latlon[b])
    def distance(i, j):
        return distances[min(i, j), max(i, j)]
    # Isolates retain themselves; no 0 * infinity score.
    score = [sum(distance(i, j) for j in sorted(adj[i])) if ranking == 'geometry'
             else len(adj[i]) if ranking == 'degree' else i for i in range(n)]
    kept = set()
    excluded = set()
    for i in sorted(range(n), key=lambda i: (score[i], i)):
        if i not in excluded:
            kept.add(i)
            excluded.update(adj[i])
    reps = sorted(kept)
    coarse_id = {r: k for k, r in enumerate(reps)}
    parent_rep = []
    for i in range(n):
        if i in kept:
            parent_rep.append(i)
        else:
            candidates = adj[i] & kept
            if not candidates:
                raise RuntimeError('undirected maximal-set invariant failed')
            parent_rep.append(min(candidates, key=lambda j: (distance(i, j), j)))
    parent = [coarse_id[r] for r in parent_rep]
    area = [0.] * len(reps)
    for i, p in enumerate(parent):
        area[p] += g.area[i]
    if not all(math.isfinite(a) for a in area):
        raise ValueError('coarse area overflow')
    edges = {(min(parent[a], parent[b]), max(parent[a], parent[b]))
             for a, b in g.edges if parent[a] != parent[b]}
    coarse = SphereGraph([g.latlon[r] for r in reps], area, edges)
    audit = {'ranking': ranking, 'representatives': reps, 'score': score,
             'parent_representative': parent_rep,
             'max_parent_angle': max((distance(i, r) for i, r in enumerate(parent_rep) if i != r), default=0.),
             'isolated_nodes': sum(not s for s in adj)}
    return coarse, parent, audit
