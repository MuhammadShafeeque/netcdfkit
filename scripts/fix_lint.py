"""Fix common linting issues in the codebase."""

from pathlib import Path


def replace_tuple_isinstance(content: str) -> str:
    """Replace tuple-based isinstance checks with union syntax."""
    replacements = [
        ("(int, np.integer)", "int | np.integer"),
        (
            "(np.integer, np.int8, np.int16, np.int32, np.int64)",
            "np.integer | np.int8 | np.int16 | np.int32 | np.int64",
        ),
        (
            "(np.floating, np.float16, np.float32, np.float64)",
            "np.floating | np.float16 | np.float32 | np.float64",
        ),
        ("(list, dict, str)", "list | dict | str"),
    ]

    for old, new in replacements:
        content = content.replace(old, new)
    return content


def fix_unused_loop_vars(content: str) -> str:
    """Fix unused loop variables by prefixing them with underscore."""
    replacements = [
        ("for idx, (_, point)", "for _idx, (_, point)"),
        ("for country_code, summary", "for country_code, _summary"),
        ("for i, line", "for _, line"),
    ]

    for old, new in replacements:
        content = content.replace(old, new)
    return content


def fix_mutable_defaults(content: str) -> str:
    """Fix mutable default arguments."""
    content = content.replace(
        "statistics: list[str] = [\"mean\"]",
        "statistics: list[str] | None = None",
    )
    return content


def fix_grid_vars(content: str) -> str:
    """Fix uppercase grid variable names."""
    replacements = [
        ("X, Y = np.meshgrid", "x_grid, y_grid = np.meshgrid"),
        ("X[mask]", "x_grid[mask]"),
        ("Y[mask]", "y_grid[mask]"),
        ("transform, X, Y", "transform, x_grid, y_grid"),
        ("for x, y in zip(X, Y)", "for x, y in zip(x_grid, y_grid)"),
    ]

    for old, new in replacements:
        content = content.replace(old, new)
    return content


def fix_long_lines(content: str) -> str:
    """Fix lines that are too long."""
    replacements = [
        (
            "f\"    Splitting large cluster ({len(cluster_points)} points) into {n_sub_chunks} sub-chunks\"",
            "f\"    Splitting cluster {len(cluster_points)} pts into {n_sub_chunks} parts\"",
        ),
        (
            "f\"    ✓ Point {point_idx}: Extracted {len(timeseries)} timesteps, {n_valid} valid values\"",
            "f\"    ✓ Pt {point_idx}: {len(timeseries)} steps, {n_valid} valid\"",
        ),
    ]

    for old, new in replacements:
        content = content.replace(old, new)
    return content


def remove_unused_vars(content: str) -> str:
    """Remove unused variable assignments."""
    lines = content.splitlines()
    output = []
    skip_next = False

    for _, line in enumerate(lines):
        if skip_next:
            skip_next = False
            continue

        unused_vars = [
            "bounds = chunk[\"bounds\"]",
            "dx = x_coords[1]",
            "dy = y_coords[1]",
        ]
        if any(var in line for var in unused_vars):
            skip_next = True  # Skip next empty line
            continue

        output.append(line)

    return "\n".join(output)


def fix_file(file_path: Path) -> None:
    """Apply all fixes to a file."""
    print(f"Fixing {file_path}")

    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    # Apply fixes
    content = replace_tuple_isinstance(content)
    content = fix_unused_loop_vars(content)
    content = fix_mutable_defaults(content)
    content = fix_grid_vars(content)
    content = fix_long_lines(content)
    content = remove_unused_vars(content)

    # Write back
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Fixed {file_path}")


def main() -> None:
    """Fix all Python files in src directory."""
    base_dir = Path("src/netcdfkit")
    for file_path in base_dir.glob("*.py"):
        fix_file(file_path)


if __name__ == "__main__":
    main()
