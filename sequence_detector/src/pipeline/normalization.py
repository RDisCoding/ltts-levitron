import unicodedata
from typing import Any, Dict, List
from src.utils.logger import logger

class TableNormalization:
    """Normalizes the extracted Sequence of Operations matrix."""

    def normalize_table(self, table: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        """
        Takes the raw Azure table and converts it into a 2D array of normalized cell data.
        Handles merged cells by copying the content to all spanned cells.
        """
        row_count = table.get("rowCount", 0)
        column_count = table.get("columnCount", 0)
        cells = table.get("cells", [])

        # Initialize an empty 2D grid
        grid = [[{"content": "", "is_header": False, "is_marker": False, "marker_value": False} 
                 for _ in range(column_count)] for _ in range(row_count)]

        logger.info(f"Normalizing table with {row_count} rows and {column_count} columns.")

        for cell in cells:
            row_idx = cell.get("rowIndex", 0)
            col_idx = cell.get("columnIndex", 0)
            row_span = cell.get("rowSpan", 1)
            col_span = cell.get("columnSpan", 1)
            
            raw_content = cell.get("content", "")
            is_header = cell.get("kind") == "columnHeader" or cell.get("kind") == "rowHeader"
            
            normalized_text = self._normalize_text(raw_content)
            is_marker, marker_value = self._evaluate_marker(normalized_text)

            cell_data = {
                "content": normalized_text,
                "is_header": is_header,
                "is_marker": is_marker,
                "marker_value": marker_value
            }

            # Fill all cells covered by the span (preserves merged cell relationships)
            for r in range(row_idx, row_idx + row_span):
                for c in range(col_idx, col_idx + col_span):
                    if r < row_count and c < column_count:
                        grid[r][c] = cell_data

        return grid

    def _normalize_text(self, text: str) -> str:
        """
        Trims whitespace, merges wrapped text (removes newlines), and normalizes Unicode.
        """
        if not text:
            return ""
        
        # Normalize unicode characters
        text = unicodedata.normalize("NFKC", text)
        
        # Replace newlines with spaces to merge wrapped text
        text = text.replace('\n', ' ').replace('\r', '')
        
        # Trim leading/trailing whitespace and reduce multiple spaces to one
        text = " ".join(text.split())
        return text

    def _evaluate_marker(self, text: str) -> tuple[bool, bool]:
        """
        Evaluates if the cell text is a decision marker and its boolean value.
        Returns a tuple: (is_marker, marker_value)
        """
        if not text:
            # Empty cells are evaluated as FALSE
            return True, False

        # If the text is very short (1-3 characters), it's highly likely a marker like "X", "Y", "Yes", bullet
        text_lower = text.lower()
        positive_markers = {"x", "y", "yes", "true", "1", "•", "*", "✓"}
        negative_markers = {"n", "no", "false", "0", "-", "n/a"}

        if text_lower in positive_markers:
            return True, True
        elif text_lower in negative_markers:
            return True, False
            
        # If it's short and not obviously negative, treat it as a positive marker (e.g. an obscure bullet point character)
        if len(text) <= 2:
            return True, True

        # Not a marker (it's regular text like a room name or device name)
        return False, False
