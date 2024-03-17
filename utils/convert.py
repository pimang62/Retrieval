from typing import Any, Dict, List, Optional
import numpy as np

def convert_to_numpy(embed: List[List[float]]) -> np.ndarray:
    """Convert to numpy as faiss type"""
    return np.array(embed).astype(np.float32)