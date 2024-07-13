import numpy as np

def generate_cell_bounds(cell_counts):
    cell_bounds = {}
    start = 0
    for cell_type, count in cell_counts.items():
        end = start + count
        cell_bounds[cell_type] = (start, end)
        start = end
    return cell_bounds
