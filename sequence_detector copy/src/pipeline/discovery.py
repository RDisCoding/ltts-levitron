from typing import Any, Dict, List, Optional
from src.utils.logger import logger

class TableDiscovery:
    """Discovers and extracts the Sequence of Operations matrix from Document Intelligence output."""
    
    # Keywords that often indicate a Sequence of Operations table
    ENGINEERING_KEYWORDS = {"sequence", "operation", "control", "device", "point", "equipment"}
    ROOM_KEYWORDS = {"room", "space", "area", "zone"}
    DEVICE_KEYWORDS = {"sensor", "thermostat", "switch", "relay", "valve", "damper", "contact"}

    def find_target_table(self, analyze_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Searches all detected tables and returns the most probable Sequence of Operations matrix.
        """
        tables = analyze_result.get("tables", [])
        logger.info(f"Discovered {len(tables)} tables in the document.")
        
        if not tables:
            logger.warning("No tables were found in the document.")
            return None

        candidate_tables = []

        for i, table in enumerate(tables):
            score = self._score_table(table)
            logger.debug(f"Table {i} scored: {score}")
            
            if score > 0:
                candidate_tables.append((score, table, i))

        if not candidate_tables:
            logger.warning("No tables matched the characteristics of a Sequence of Operations matrix.")
            return None

        # Sort by score descending
        candidate_tables.sort(key=lambda x: x[0], reverse=True)
        best_score, best_table, best_index = candidate_tables[0]
        
        logger.info(f"Selected table index {best_index} as the Sequence of Operations matrix (Score: {best_score}).")
        return best_table

    def _score_table(self, table: Dict[str, Any]) -> int:
        """
        Heuristically scores a table based on its text content to determine if it's the target matrix.
        """
        score = 0
        cells = table.get("cells", [])
        
        # We look for specific keywords inside the cells
        text_corpus = " ".join(cell.get("content", "").lower() for cell in cells)

        # Check for engineering terminology
        if any(keyword in text_corpus for keyword in self.ENGINEERING_KEYWORDS):
            score += 2

        # Check for room identifiers
        if any(keyword in text_corpus for keyword in self.ROOM_KEYWORDS):
            score += 2

        # Check for control device names
        if any(keyword in text_corpus for keyword in self.DEVICE_KEYWORDS):
            score += 2

        # Check for decision matrix layout (usually many columns indicating different devices/actions)
        column_count = table.get("columnCount", 0)
        row_count = table.get("rowCount", 0)
        
        if column_count >= 3 and row_count >= 2:
            score += 3  # High probability it's a matrix if it has sufficient dimensions

        return score
