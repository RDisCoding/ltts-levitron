from typing import Dict
from src.models.domain import Room, DeviceRequirement, Rule, RuleCollection
from src.utils.logger import logger

class RuleGenerator:
    """Converts a boolean decision matrix into structured engineering rules."""

    def generate_rules(self, matrix_dict: Dict[str, Dict[str, bool]]) -> RuleCollection:
        """
        Takes the raw matrix dictionary and maps it to strongly typed Pydantic models.
        Only positive requirements (True) are typically actioned downstream, but we 
        capture both states in the requirements list for completeness.
        """
        logger.info("Generating engineering rules from decision matrix.")
        
        rules_list = []
        
        for room_name, devices_map in matrix_dict.items():
            if not room_name.strip():
                continue
                
            # Create the Room model
            room_model = Room(name=room_name.strip())
            
            # Create the list of DeviceRequirements for this room
            requirements = []
            for device_name, is_required in devices_map.items():
                if not device_name.strip():
                    continue
                    
                req_model = DeviceRequirement(
                    device_name=device_name.strip(),
                    is_required=is_required
                )
                requirements.append(req_model)
                
            # Create the Rule model linking the room to its requirements
            rule_model = Rule(
                room=room_model,
                requirements=requirements
            )
            rules_list.append(rule_model)

        logger.info(f"Generated {len(rules_list)} rules successfully.")
        return RuleCollection(rules=rules_list)
