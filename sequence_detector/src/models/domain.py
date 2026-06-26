from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class Room(BaseModel):
    name: str = Field(..., description="The name or identifier of the room/space.")

class DeviceRequirement(BaseModel):
    device_name: str = Field(..., description="The name of the control device or action.")
    is_required: bool = Field(..., description="Whether this device is required for the room.")

class DecisionMatrix(BaseModel):
    matrix: Dict[str, Dict[str, bool]] = Field(..., description="The raw True/False mapping of Rooms to Devices.")

class Rule(BaseModel):
    room: Room
    requirements: List[DeviceRequirement] = Field(default_factory=list, description="List of all devices and their requirement status for this room.")

class RuleCollection(BaseModel):
    rules: List[Rule] = Field(default_factory=list, description="Collection of all engineering rules derived from the document.")

class ProcessingMetadata(BaseModel):
    source_file: str
    tables_discovered: int
    selected_table_index: Optional[int]
    rows_processed: int
    columns_processed: int
    normalization_actions: List[str] = Field(default_factory=list)

class ValidationReport(BaseModel):
    warnings: List[str] = Field(default_factory=list, description="Non-fatal issues discovered during generation (e.g. empty rows).")
    errors: List[str] = Field(default_factory=list, description="Fatal issues that violate constraints.")
    is_valid: bool = Field(True, description="False if any errors are present.")
