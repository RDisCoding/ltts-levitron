import pytest
from src.pipeline.normalization import TableNormalization
from src.pipeline.matrix_builder import DecisionMatrixBuilder
from src.pipeline.rule_generator import RuleGenerator
from src.pipeline.validation import PipelineValidator
from src.models.domain import RuleCollection, Rule, Room, DeviceRequirement

class TestPipeline:
    
    @pytest.fixture
    def mock_azure_table(self):
        # A simplified representation of what Azure might return
        return {
            "rowCount": 3,
            "columnCount": 3,
            "cells": [
                {"rowIndex": 0, "columnIndex": 0, "content": "Room Name", "kind": "columnHeader"},
                {"rowIndex": 0, "columnIndex": 1, "content": "Thermostat", "kind": "columnHeader"},
                {"rowIndex": 0, "columnIndex": 2, "content": "Sensor", "kind": "columnHeader"},
                {"rowIndex": 1, "columnIndex": 0, "content": "Office"},
                {"rowIndex": 1, "columnIndex": 1, "content": "X"},
                {"rowIndex": 1, "columnIndex": 2, "content": ""},
                {"rowIndex": 2, "columnIndex": 0, "content": "Lobby"},
                {"rowIndex": 2, "columnIndex": 1, "content": "False"},
                {"rowIndex": 2, "columnIndex": 2, "content": "Y"}
            ]
        }

    def test_marker_normalization(self):
        norm = TableNormalization()
        
        # Test True markers
        assert norm._evaluate_marker("X") == (True, True)
        assert norm._evaluate_marker("Yes") == (True, True)
        assert norm._evaluate_marker("•") == (True, True)
        
        # Test False markers
        assert norm._evaluate_marker("No") == (True, False)
        assert norm._evaluate_marker("") == (True, False)
        
        # Test Regular text
        assert norm._evaluate_marker("Conference Room") == (False, False)

    def test_table_normalization(self, mock_azure_table):
        norm = TableNormalization()
        grid = norm.normalize_table(mock_azure_table)
        
        assert len(grid) == 3
        assert len(grid[0]) == 3
        
        # Check office thermostat is correctly parsed as marker=True, value=True
        assert grid[1][1]["is_marker"] is True
        assert grid[1][1]["marker_value"] is True
        
        # Check office sensor is correctly parsed as marker=True, value=False (empty string)
        assert grid[1][2]["is_marker"] is True
        assert grid[1][2]["marker_value"] is False

    def test_decision_matrix_generation(self, mock_azure_table):
        norm = TableNormalization()
        grid = norm.normalize_table(mock_azure_table)
        
        builder = DecisionMatrixBuilder()
        matrix = builder.build_matrix(grid)
        
        assert "Office" in matrix
        assert matrix["Office"]["Thermostat"] is True
        assert matrix["Office"]["Sensor"] is False
        
        assert "Lobby" in matrix
        assert matrix["Lobby"]["Thermostat"] is False
        assert matrix["Lobby"]["Sensor"] is True

    def test_rule_generation(self):
        matrix_dict = {
            "Office": {"Thermostat": True, "Sensor": False}
        }
        
        gen = RuleGenerator()
        collection = gen.generate_rules(matrix_dict)
        
        assert isinstance(collection, RuleCollection)
        assert len(collection.rules) == 1
        assert collection.rules[0].room.name == "Office"
        
        reqs = collection.rules[0].requirements
        assert len(reqs) == 2
        assert reqs[0].device_name == "Thermostat"
        assert reqs[0].is_required is True

    def test_validation(self):
        # Create a rule collection with an empty room name to trigger an error
        invalid_rules = RuleCollection(rules=[
            Rule(room=Room(name=""), requirements=[DeviceRequirement(device_name="Sensor", is_required=True)])
        ])
        
        validator = PipelineValidator()
        report = validator.validate_rules(invalid_rules)
        
        assert report.is_valid is False
        assert len(report.errors) == 1
        assert "empty room identifier" in report.errors[0]
