from dataclasses import dataclass
from typing import List, Tuple

# Offsets are (0, -1) e.g.
Offset = Tuple[int, int]

# Simple Dataclass to pass between processes
@dataclass
class ScoredFootprint:
    shot_number: str
    beam: str
    delta_time: float
    offset_scores: List[Tuple[Offset, float]]
    cluster_n: int