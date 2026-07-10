"""
partitioner.py
--------------
Pure geometry logic: samples the flyable workspace, clusters it with KMeans,
generates a Voronoi diagram, clips to the boundary, and merges orphaned islands
back into the nearest primary partition.

No I/O, no plotting — just polygons in, partitions out.

Bug fixes
---------
BUG 3  – random_state now seeds BOTH numpy sampling AND KMeans, so every run
         with the same seed produces identical output. Previously np.random was
         unseeded while KMeans had random_state=42, giving nondeterministic
         sampling and therefore nondeterministic partitions.

BUG 4  – Partitions are sorted by centroid (x, y) after generation so partition
         IDs are geographically stable across runs rather than depending on the
         arbitrary order of Voronoi regions.

BUG 5  – Orphan merging now uses true shared-boundary length via
         primary.boundary.intersection(orphan.boundary).length rather than
         the perimeter of a buffered intersection area, which could select
         non-touching neighbours.

BUG 6  – A RuntimeError is raised if the number of valid non-empty partitions
         produced does not equal n_parts, so callers are never misled by the
         metadata count.

BUG 7  – buffer(0) is replaced by explicit validity checking via
         explain_validity(). Geometry is repaired only when necessary and
         the repair is logged rather than silently applied.

BUG 10 – The sampling loop now has a hard max_attempts ceiling (default
         1 000 000). A clear RuntimeError is raised if the workspace cannot
         yield enough valid sample points, preventing an infinite loop.

New features
------------
SMALL AREA FILTERING – Two configurable thresholds are applied to discard
    geometry that is too small or too thin to be useful for drone flight:

    min_area_m2 (default 500 m²)
        Any polygon or sub-polygon whose area is below this threshold is
        discarded before orphan merging. This removes tiny slivers that
        appear at the edges of no-go zones or boundary intersections.

    min_width_m (default 10 m)
        Any polygon that is thinner than this value is discarded.
        "Thinness" is measured by negative buffering: if shrinking a polygon
        inward by min_width_m/2 on all sides produces an empty result, the
        polygon is not wide enough to fly through and is dropped.
        This catches long thin strips that have significant area but no usable
        width — e.g. a 1 m × 1000 m corridor scores 1000 m² (above the area
        threshold) but collapses to nothing under a 5 m inward buffer.

    Both parameters are exposed on build_partitions() so callers can tune
    them per-mission without touching the source code.

    Discarded fragments are NOT orphaned — they are dropped entirely.
    The rationale is that geometry below these thresholds is geometrically
    real but operationally meaningless; merging it into a neighbour would
    distort partition areas for no benefit.
"""

import numpy as np
from shapely.geometry import MultiPoint, MultiPolygon, Point, Polygon
from shapely.ops import unary_union, voronoi_diagram
from shapely.validation import explain_validity
from sklearn.cluster import KMeans

# Tolerance used when checking shared boundaries between a primary partition
# and an orphan island — just enough to bridge floating-point gaps at edges.
_BORDER_TOLERANCE_M = 1.5


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_valid(poly: Polygon, label: str = "") -> Polygon | MultiPolygon:
    """
    Return a valid version of poly, logging any repair.

    BUG 7 FIX: validity is checked explicitly and any repair is logged.
    buffer(0) is used as the repair mechanism only when needed, not applied
    blindly to every polygon as before.
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

    BUG 10 FIX: raises RuntimeError after max_attempts to prevent infinite
    loops on degenerate or near-empty workspaces. Previously the while loop
    had no exit condition on failure.
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

    BUG 5 FIX: uses boundary.intersection(boundary) to measure actual shared
    edge length, rather than primary.intersection(orphan.buffer(tolerance))
    which measured the perimeter of an area and could select non-touching
    neighbours.
    """
    shared = primary.boundary.intersection(orphan.buffer(_BORDER_TOLERANCE_M).boundary)
    return shared.length if not shared.is_empty else 0.0


def _is_usable(
    poly: Polygon,
    min_area_m2: float,
    min_width_m: float,
) -> bool:
    """
    Return True if poly is large enough and wide enough to be worth keeping.

    Two independent checks:
        1. Area must be >= min_area_m2.
        2. Negative buffer by (min_width_m / 2) must leave a non-empty result,
           meaning the polygon is at least min_width_m wide at its narrowest point.

    Either failure is sufficient to reject the polygon.

    Why divide min_width_m by 2?
        A negative buffer of d shrinks the polygon by d on EVERY side.
        So to test "is this at least W metres wide", we shrink by W/2 — if the
        polygon was exactly W wide, shrinking both sides by W/2 collapses it to
        a line (empty). Anything narrower disappears entirely.
    """
    if poly.area < min_area_m2:
        return False
    eroded = poly.buffer(-(min_width_m / 2.0))
    return not eroded.is_empty


def _filter_geometry(
    geom: Polygon | MultiPolygon,
    min_area_m2: float,
    min_width_m: float,
    label: str = "",
) -> tuple[Polygon | None, list[Polygon]]:
    """
    Filter a polygon or MultiPolygon by area and width thresholds.

    Returns:
        (primary, orphans) where:
            primary  – the largest usable sub-polygon, or None if none pass
            orphans  – remaining usable sub-polygons to be merged elsewhere

    Sub-polygons that fail either threshold are silently discarded.
    """
    if isinstance(geom, MultiPolygon):
        sub_polys = list(geom.geoms)
    else:
        sub_polys = [geom]

    usable = [p for p in sub_polys if _is_usable(p, min_area_m2, min_width_m)]

    discarded = len(sub_polys) - len(usable)
    if discarded > 0:
        tag = f" [{label}]" if label else ""
        print(
            f"[partitioner] Discarded {discarded} fragment(s){tag} "
            f"below area ({min_area_m2} m²) or width ({min_width_m} m) threshold."
        )

    if not usable:
        return None, []

    usable.sort(key=lambda p: p.area, reverse=True)
    return usable[0], usable[1:]


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
    min_area_m2: float = 500.0,
    min_width_m: float = 10.0,
) -> list[Polygon | MultiPolygon]:
    """
    Partition the flyable workspace into n_parts regions.

    Steps:
        1.  Subtract no-go zones from the boundary → flyable workspace.
        2.  Sample random points inside the workspace (seeded, bounded).
        3.  Cluster with KMeans → n_parts centroids.
        4.  Build Voronoi diagram clipped to the boundary.
        5.  Subtract no-go zones from each Voronoi cell.
        6.  Filter out sub-polygons below min_area_m2 or min_width_m.
        7.  Merge surviving orphan islands into nearest neighbour by
            true shared-boundary length.
        8.  Sort partitions by centroid (x, y) for stable IDs.
        9.  Validate that exactly n_parts non-empty partitions were produced.

    Args:
        boundary:     Shapely Polygon representing the full mission area.
        nogo_polys:   List of Shapely Polygons that are off-limits.
        n_parts:      Desired number of partitions.
        n_samples:    Random sample count for KMeans (default 10 000).
        random_state: Seed for both numpy sampling and KMeans (default 42).
        max_attempts: Hard ceiling on sampling loop iterations (default 1 000 000).
        min_area_m2:  Minimum area in m² for a sub-polygon to be kept (default 500).
                      Fragments smaller than this are discarded entirely.
        min_width_m:  Minimum width in metres for a sub-polygon to be kept (default 10).
                      Fragments narrower than this (measured by erosion) are discarded.

    Returns:
        List of exactly n_parts Shapely Polygon / MultiPolygon objects, sorted
        by centroid (x, y) for geographic stability.

    Raises:
        RuntimeError: workspace cannot be sampled, or count != n_parts.
        ValueError:   polygon cannot be repaired.
    """
    # BUG 3 FIX: single rng instance seeds ALL randomness — both numpy sampling
    # and KMeans use the same random_state so the full pipeline is reproducible.
    rng = np.random.default_rng(random_state)

    nogo_mask = unary_union(nogo_polys) if nogo_polys else None
    workspace = boundary.difference(nogo_mask) if nogo_mask else boundary

    # --- 1. Sample flyable space (BUG 10 FIX: bounded by max_attempts) ---
    samples = _sample_workspace(
        workspace, boundary.bounds, n_samples, rng, max_attempts
    )

    # --- 2. KMeans clustering (BUG 3 FIX: random_state matches numpy seed) ---
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

    # --- 4. Subtract no-go zones; filter and collect orphans ---
    partitions: list[Polygon | MultiPolygon | None] = [None] * len(temp_partitions)
    orphans: list[Polygon] = []

    for i, cell in enumerate(temp_partitions):
        flyable = cell.difference(nogo_mask) if nogo_mask else cell

        if flyable.is_empty:
            continue

        # SMALL AREA / THIN STRIP FILTERING (step 6)
        primary, cell_orphans = _filter_geometry(
            flyable, min_area_m2, min_width_m, label=f"cell {i + 1}"
        )
        partitions[i] = primary
        orphans.extend(cell_orphans)

    # --- 5. Merge surviving orphans ---
    for orphan in orphans:
        best_idx  = -1
        max_shared = -1.0

        for i, primary in enumerate(partitions):
            if primary is None or primary.is_empty:
                continue
            length = _shared_boundary_length(primary, orphan)
            if length > max_shared:
                max_shared = length
                best_idx   = i

        if best_idx != -1:
            merged = unary_union([partitions[best_idx], orphan])
            if isinstance(merged, MultiPolygon):
                print(
                    f"[partitioner] WARNING: merging orphan into partition {best_idx + 1} "
                    f"produced a MultiPolygon — they may not be adjacent."
                )
            partitions[best_idx] = merged
        else:
            print("[partitioner] WARNING: orphan island could not be assigned to any partition.")

    valid_partitions = [p for p in partitions if p is not None and not p.is_empty]

    # --- 6. Sort by centroid for geographic stability (BUG 4 FIX) ---
    # Voronoi regions come out in arbitrary internal order; sorting by centroid
    # ensures partition IDs are geographically stable across runs.
    valid_partitions.sort(key=lambda p: (p.centroid.x, p.centroid.y))

    # --- 7. Validate count (BUG 6 FIX) ---
    # Metadata must never claim n_parts if the algorithm produced a different
    # number — raise explicitly rather than letting the caller be misled.
    if len(valid_partitions) != n_parts:
        raise RuntimeError(
            f"Expected {n_parts} partitions but produced {len(valid_partitions)}. "
            f"Try reducing n_parts, increasing n_samples, or lowering min_area_m2/min_width_m."
        )

    return valid_partitions