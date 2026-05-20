"""
Re-export from rina package for backward compatibility.
Old scripts that do `from modules.temporal_snn_cell import ...` still work.
"""
from rina import TemporalSNNCell, TemporalSNNModel
