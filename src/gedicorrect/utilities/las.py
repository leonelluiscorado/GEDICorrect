"""LAS/LAZ preparation helpers."""

from pathlib import Path

import laspy


def convert_laz_directory(las_dir):
    """Convert every LAZ file below a directory to an adjacent LAS file."""

    input_path = Path(las_dir).expanduser().resolve()
    if not input_path.is_dir():
        raise ValueError(f"LAS directory does not exist: {input_path}")

    laz_files = sorted(input_path.rglob("*.laz"))
    if not laz_files:
        raise ValueError(f"No .laz files found in {input_path}")

    output_files = []
    for laz_file in laz_files:
        output_file = laz_file.with_suffix(".las")
        print(f"Converting {laz_file} to {output_file}")
        las = laspy.convert(laspy.read(laz_file))
        las.write(output_file)
        output_files.append(str(output_file))
    return output_files
