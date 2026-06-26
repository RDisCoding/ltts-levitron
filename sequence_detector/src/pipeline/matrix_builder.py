from typing import Any, Dict, List
from src.utils.logger import logger

class DecisionMatrixBuilder:
    """Transforms the normalized grid into a structured boolean decision matrix."""

    def build_matrix(self, grid: List[List[Dict[str, Any]]]) -> Dict[str, Dict[str, bool]]:
        """
        Dynamically extracts devices (columns) and rooms (rows) to build a mapping:
        { "Room A": { "Device X": True, "Device Y": False }, ... }
        
        Assumes the first column contains the primary subjects (Rooms) and the top rows 
        contain the properties (Devices/Actions).
        """
        logger.info("Starting decision matrix construction from normalized grid.")
        
        if not grid or not grid[0]:
            logger.warning("Empty grid provided to DecisionMatrixBuilder.")
            return {}

        row_count = len(grid)
        col_count = len(grid[0])

        # Step 1: Discover boundaries
        # We need to find where the header ends and data begins.
        # A simple heuristic: The first cell containing a boolean marker indicates the start of the data area.
        data_start_row, data_start_col = self._find_data_origin(grid)
        
        if data_start_row == -1 or data_start_col == -1:
            logger.warning("Could not find the start of the boolean data matrix. Returning empty matrix.")
            return {}
            
        logger.debug(f"Discovered data origin at Row: {data_start_row}, Col: {data_start_col}")

        # Step 2: Extract Column Headers (Devices / Actions)
        # We read upwards from the data_start_row to form composite headers.
        devices = self._extract_column_headers(grid, data_start_row, data_start_col, col_count)

        # Step 3: Extract Row Headers (Rooms) and Build Matrix
        matrix = {}
        for r in range(data_start_row, row_count):
            # Extract composite row header (e.g. if the room name is split across multiple leading columns)
            room_parts = []
            for c in range(0, data_start_col):
                content = grid[r][c]["content"]
                if content and content not in room_parts:
                    room_parts.append(content)
            
            room_name = " ".join(room_parts).strip()
            
            if not room_name:
                continue
                
            if room_name not in matrix:
                matrix[room_name] = {}
                
            # Now map the boolean values for this row
            for c in range(data_start_col, col_count):
                device_name = devices[c - data_start_col]
                # Skip if there's no device name associated with this column
                if not device_name:
                    continue
                    
                is_marker = grid[r][c]["is_marker"]
                marker_value = grid[r][c]["marker_value"]
                
                # If it's not a recognized marker, we default to False.
                is_required = marker_value if is_marker else False
                
                matrix[room_name][device_name] = is_required

        logger.info(f"Matrix built successfully: discovered {len(matrix)} unique rooms and {len(devices)} device columns.")
        return matrix

    def _find_data_origin(self, grid: List[List[Dict[str, Any]]]) -> tuple[int, int]:
        """Finds the first cell that acts as a boolean marker."""
        for r in range(len(grid)):
            for c in range(len(grid[r])):
                if grid[r][c]["is_marker"]:
                    return r, c
        return -1, -1

    def _extract_column_headers(self, grid: List[List[Dict[str, Any]]], data_start_row: int, data_start_col: int, col_count: int) -> List[str]:
        """Extracts column headers by reading above the data start row."""
        devices = []
        for c in range(data_start_col, col_count):
            header_parts = []
            for r in range(0, data_start_row):
                content = grid[r][c]["content"]
                # Avoid duplicating vertically merged headers
                if content and content not in header_parts:
                    header_parts.append(content)
            devices.append(" ".join(header_parts).strip())
        return devices
