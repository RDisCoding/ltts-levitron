from typing import Dict, List
from src.models.domain import RuleCollection, ValidationReport
from src.utils.logger import logger

class PipelineValidator:
    """Validates the generated rules and detects any anomalies."""

    def validate_rules(self, rule_collection: RuleCollection) -> ValidationReport:
        """
        Validates the generated rule collection.
        Checks for duplicate rooms, empty identifiers, and malformed structures.
        Returns a ValidationReport with warnings and errors.
        """
        logger.info("Running validation on generated rules.")
        
        warnings: List[str] = []
        errors: List[str] = []
        seen_rooms: set = set()

        if not rule_collection.rules:
            errors.append("Rule collection is completely empty.")

        for rule in rule_collection.rules:
            room_name = rule.room.name
            
            if not room_name:
                errors.append("Found rule with an empty room identifier.")
                continue

            if room_name in seen_rooms:
                errors.append(f"Duplicate room identifier detected: '{room_name}'.")
            else:
                seen_rooms.add(room_name)

            if not rule.requirements:
                warnings.append(f"Room '{room_name}' has no device requirements listed.")

            for req in rule.requirements:
                if not req.device_name:
                    errors.append(f"Empty device name found in room '{room_name}'.")
                    
        is_valid = len(errors) == 0
        
        if errors:
            logger.error(f"Validation failed with {len(errors)} errors.")
        if warnings:
            logger.warning(f"Validation completed with {len(warnings)} warnings.")
            
        return ValidationReport(
            warnings=warnings,
            errors=errors,
            is_valid=is_valid
        )
