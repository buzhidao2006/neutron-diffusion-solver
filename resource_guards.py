"""Sizing estimates and safety limits for high-dimensional diffusion solves."""

MAX_SPATIAL_NODES = {2: 100_000, 3: 30_000}
ESTIMATED_BYTES_PER_NODE = {2: 512, 3: 1_024}


def estimate_two_group_resources(dimension, *grid_counts):
    """Return node, unknown, and approximate working-memory requirements."""
    if dimension not in MAX_SPATIAL_NODES:
        raise ValueError("dimension must be 2 or 3.")
    spatial_nodes = 1
    for count in grid_counts:
        spatial_nodes *= count
    estimated_memory_mb = (
        spatial_nodes * ESTIMATED_BYTES_PER_NODE[dimension] / 1024 ** 2
    )
    return {
        "spatial_nodes": spatial_nodes,
        "unknowns": 2 * spatial_nodes,
        "estimated_memory_mb": estimated_memory_mb,
        "safe_node_limit": MAX_SPATIAL_NODES[dimension],
    }


def guard_problem_size(dimension, *grid_counts):
    """Reject a problem before sparse-matrix assembly exceeds the safe budget."""
    estimate = estimate_two_group_resources(dimension, *grid_counts)
    if estimate["spatial_nodes"] > estimate["safe_node_limit"]:
        shape = "×".join(str(count) for count in grid_counts)
        raise ValueError(
            f"{dimension}D grid {shape} has {estimate['spatial_nodes']:,} spatial nodes "
            f"({estimate['unknowns']:,} two-group unknowns), exceeding the safe limit of "
            f"{estimate['safe_node_limit']:,} nodes. Reduce the grid resolution before solving."
        )
    return estimate
