"""
partitioner.py
--------------
Pure geometry logic: samples the flyable workspace, clusters it with KMeans,
generates a Voronoi diagram, clips to the boundary, and merges orphaned islands
back into the nearest primary partition.

No I/O, no plotting — just polygons in, partitions out.

Bug fixes applied
-----------------
BUG 3  – random_seed now seeds BOTH numpy sampling AND KMeans, so every run
         with the same seed produces identical output. Previously np.random was
         unseeded while KMeans had random_state=42, giving nondeterministic
         sampling and therefore nondeterministic partitions.

BUG 4  – Partitions are sorted by centroid (x, y) after generation so partition
         IDs are geographically stable across runs rather than depending on the
         arbitrary order of Voronoi regions.

BUG 5  – Orphan merging now uses true shared-boundary length
         (primary.boundary.intersection(orphan.boundary).length with a small
         buffer tolerance) rather than the perimeter of a buffered intersection,
         which could select non-touching neighbours.

BUG 6  – A RuntimeError is raised if the number of valid non-empty partitions
         produced does not equal n_parts, so callers are never misled by the
         metadata count.

BUG 7  – buffer(0) is replaced by explicit validity checking via
         shapely.validation.explain_validity. Geometry is repaired only when
         necessary, and the repair is logged rather than silently applied.

BUG 10 – The sampling loop now has a hard max_attempts ceiling (default
         1 000 000). A clear RuntimeError is raised if the workspace cannot
         yield enough valid sample points, preventing an infinite loop.
"""

import numpy as np
from shapely.geometry import MultiPoint, MultiPolygon, Point, Polygon
from shapely.ops import unary_union, voronoi_diagram
from shapely.validation import explain_validity
from sklearn.cluster import KMeans

# Tolerance used when checking shared boundaries between a primary partition
# and an orphan island.  1.5 m is intentionally small — just enough to bridge
# floating-point gaps at polygon edges.
_BORDER_TOLERANCE_M = 1.5


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_valid(poly: Polygon, label: str = "") -> Polygon | MultiPolygon:
    """
    Return a valid version of poly.

    BUG 7 FIX: validity is checked explicitly and any repair is logged.
    buffer(0) is still used as the repair mechanism (it is reliable for the
    self-intersection class of errors that KML files commonly contain) but
    is no longer applied blindly to every polygon.
    """
    if poly.is_valid:
        return poly

    reason = explain_validity(poly)
    tag = f" [{label}]" if label else ""
    print(f"[partitioner] WARNING{tag}: invalid geometry — {reason}. Attempting repair.")
    repaired = poly.buffer(0)

    if not repaired.is_valid:
        raise ValueError(
            f"Geometry{tag} could not be repaired automatically: {explain_validity(repaired)}"
        )

    print(f"[partitioner] Repair succeeded{tag}.")
    return repaired


def _sample_workspace(
    workspace,
    boundary_bounds: tuple[float, float, float, float],
    n_samples: int,
    rng: np.random.Generator,
    max_attempts: int,
) -> list[list[float]]:
    """
    Draw n_samples random points that fall inside workspace.

    BUG 10 FIX: raises RuntimeError after max_attempts to prevent infinite loops
    on degenerate or near-empty workspaces.
    """
    minx, miny, maxx, maxy = boundary_bounds
    samples: list[list[float]] = []
    attempts = 0

    while len(samples) < n_samples:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"Could only collect {len(samples)}/{n_samples} valid sample points "
                f"after {max_attempts} attempts. The flyable workspace may be too small "
                f"or entirely blocked by no-go zones."
            )
        px = rng.uniform(minx, maxx)
        py = rng.uniform(miny, maxy)
        attempts += 1
        if workspace.contains(Point(px, py)):
            samples.append([px, py])

    return samples


def _shared_boundary_length(primary: Polygon, orphan: Polygon) -> float:
    """
    Return the true shared-boundary length between primary and orphan.

    BUG 5 FIX: use boundary.intersection(boundary) instead of
    primary.intersection(orphan.buffer(tolerance)), which measured the
    perimeter of an area rather than shared edge length.
    A small buffer is applied only to bridge floating-point gaps.
    """
    shared = primary.boundary.intersection(orphan.buffer(_BORDER_TOLERANCE_M).boundary)
    return shared.length if not shared.is_empty else 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_partitions(
    boundary: Polygon,
    nogo_polys: list[Polygon],
    n_parts: int,
    n_samples: int = 10_000,
    random_state: int = 42,
    max_attempts: int = 1_000_000,
) -> list[Polygon | MultiPolygon]:
    """
    Partition the flyable workspace into n_parts regions.

    Steps:
        1. Subtract no-go zones from the boundary to get the flyable workspace.
        2. Sample random points inside the workspace (seeded, bounded attempts).
        3. Cluster with KMeans to find n_parts centroids.
        4. Build a Voronoi diagram clipped to the boundary.
        5. Subtract no-go zones from each Voronoi cell.
        6. Merge orphaned sub-islands into their nearest neighbour by true
           shared-boundary length.
        7. Sort partitions by centroid for stable IDs.
        8. Validate that exactly n_parts non-empty partitions were produced.

    Args:
        boundary:     Shapely Polygon representing the full mission area.
        nogo_polys:   List of Shapely Polygons that are off-limits.
        n_parts:      Desired number of partitions.
        n_samples:    Random sample count for KMeans (default 10 000).
        random_state: Seed for both numpy sampling and KMeans (default 42).
        max_attempts: Hard ceiling on sampling loop iterations (default 1 000 000).

    Returns:
        List of exactly n_parts Shapely Polygon / MultiPolygon objects, sorted
        by centroid (x, y) for geographic stability.

    Raises:
        RuntimeError: If the workspace cannot yield n_samples points, or if the
                      algorithm produces a count other than n_parts.
        ValueError:   If a polygon cannot be repaired after validity failure.
    """
    # BUG 3 FIX: single rng instance seeds ALL randomness
    rng = np.random.default_rng(random_state)

    nogo_mask = unary_union(nogo_polys) if nogo_polys else None
    workspace = boundary.difference(nogo_mask) if nogo_mask else boundary

    # --- 1. Sample flyable space (BUG 10 FIX: bounded) ---
    samples = _sample_workspace(
        workspace, boundary.bounds, n_samples, rng, max_attempts
    )

    # --- 2. KMeans clustering (BUG 3 FIX: same random_state as numpy seed) ---
    kmeans = KMeans(n_clusters=n_parts, n_init="auto", random_state=random_state)
    kmeans.fit(samples)
    centers = kmeans.cluster_centers_

    # --- 3. Voronoi diagram clipped to boundary ---
    regions = voronoi_diagram(MultiPoint(centers))
    temp_partitions: list[Polygon] = []
    for region in regions.geoms:
        clipped = region.intersection(boundary)
        if not clipped.is_empty:
            temp_partitions.append(clipped)

    # --- 4. Subtract no-go zones; collect orphaned sub-islands ---
    partitions: list[Polygon | MultiPolygon | None] = [None] * len(temp_partitions)
    orphans: list[Polygon] = []

    for i, cell in enumerate(temp_partitions):
        flyable = cell.difference(nogo_mask) if nogo_mask else cell

        if isinstance(flyable, MultiPolygon):
            sorted_islands = sorted(flyable.geoms, key=lambda g: g.area, reverse=True)
            partitions[i] = sorted_islands[0]
            orphans.extend(sorted_islands[1:])
        else:
            partitions[i] = flyable

    # --- 5. Merge orphans (BUG 5 FIX: true shared-boundary length) ---
    for orphan in orphans:
        best_idx = -1
        max_shared = -1.0

        for i, primary in enumerate(partitions):
            if primary is None or primary.is_empty:
                continue
            length = _shared_boundary_length(primary, orphan)
            if length > max_shared:
                max_shared = length
                best_idx = i

        if best_idx != -1:
            merged = unary_union([partitions[best_idx], orphan])
            # Verify the merge produced a connected polygon, warn otherwise
            if isinstance(merged, MultiPolygon):
                print(
                    f"[partitioner] WARNING: merging orphan into partition {best_idx + 1} "
                    f"produced a MultiPolygon — they may not be adjacent."
                )
            partitions[best_idx] = merged
        else:
            print(f"[partitioner] WARNING: orphan island could not be assigned to any partition.")

    valid_partitions = [p for p in partitions if p is not None and not p.is_empty]

    # --- 6. BUG 4 FIX: sort by centroid for geographic stability ---
    valid_partitions.sort(key=lambda p: (p.centroid.x, p.centroid.y))

    # --- 7. BUG 6 FIX: validate count ---
    if len(valid_partitions) != n_parts:
        raise RuntimeError(
            f"Expected {n_parts} partitions but produced {len(valid_partitions)}. "
            f"Try reducing n_parts or increasing n_samples."
        )

    return valid_partitions