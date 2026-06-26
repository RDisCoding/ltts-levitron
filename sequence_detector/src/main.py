import os
import json
from src.pipeline.ingestion import DocumentIngestionClient
from src.pipeline.discovery import TableDiscovery
from src.pipeline.normalization import TableNormalization
from src.pipeline.matrix_builder import DecisionMatrixBuilder
from src.pipeline.rule_generator import RuleGenerator
from src.pipeline.validation import PipelineValidator
from src.models.domain import ProcessingMetadata
from src.utils.logger import logger

def main(pdf_path: str, output_dir: str):
    logger.info("Starting Sequence of Operations Pipeline")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize components
    ingestion = DocumentIngestionClient()
    discovery = TableDiscovery()
    normalization = TableNormalization()
    matrix_builder = DecisionMatrixBuilder()
    rule_generator = RuleGenerator()
    validator = PipelineValidator()

    # Stage 1: Ingestion
    logger.info("STAGE 1: Ingestion")
    analyze_result = ingestion.analyze_document(pdf_path)
    
    # Stage 2: Table Discovery
    logger.info("STAGE 2: Table Discovery")
    target_table = discovery.find_target_table(analyze_result)
    
    if not target_table:
        logger.error("Pipeline terminated: No valid matrix table discovered.")
        return

    # Stage 3: Normalization
    logger.info("STAGE 3: Normalization")
    grid = normalization.normalize_table(target_table)

    # Stage 4: Decision Matrix
    logger.info("STAGE 4: Decision Matrix")
    matrix_dict = matrix_builder.build_matrix(grid)
    
    # Save Decision Matrix output
    matrix_out_path = os.path.join(output_dir, "decision_matrix.json")
    with open(matrix_out_path, "w", encoding="utf-8") as f:
        json.dump({"matrix": matrix_dict}, f, indent=2)

    # Stage 5: Rule Generation
    logger.info("STAGE 5: Rule Generation")
    rule_collection = rule_generator.generate_rules(matrix_dict)
    
    # Save Engineering Rules output
    rules_out_path = os.path.join(output_dir, "engineering_rules.json")
    with open(rules_out_path, "w", encoding="utf-8") as f:
        f.write(rule_collection.model_dump_json(indent=2))

    # Stage 6: Validation
    logger.info("STAGE 6: Validation")
    validation_report = validator.validate_rules(rule_collection)
    
    # Save Validation Report output
    val_out_path = os.path.join(output_dir, "validation_report.json")
    with open(val_out_path, "w", encoding="utf-8") as f:
        f.write(validation_report.model_dump_json(indent=2))

    # Generate and Save Processing Metadata
    logger.info("Generating Processing Metadata")
    total_tables = len(analyze_result.get("tables", []))
    metadata = ProcessingMetadata(
        source_file=pdf_path,
        tables_discovered=total_tables,
        selected_table_index=None, # In a fuller implementation, discovery could return this
        rows_processed=target_table.get("rowCount", 0),
        columns_processed=target_table.get("columnCount", 0),
        normalization_actions=["Trim Whitespace", "Normalize Unicode", "Evaluate Boolean Markers"]
    )
    
    meta_out_path = os.path.join(output_dir, "processing_metadata.json")
    with open(meta_out_path, "w", encoding="utf-8") as f:
        f.write(metadata.model_dump_json(indent=2))

    if not validation_report.is_valid:
        logger.error("Pipeline completed with VALIDATION ERRORS. Check validation_report.json")
    else:
        logger.info("Pipeline completed SUCCESSFULLY.")

if __name__ == "__main__":
    # Standard inputs for this phase
    INPUT_PDF = "input/Sequence Of Operations.pdf"
    OUTPUT_DIR = "output"
    
    try:
        main(INPUT_PDF, OUTPUT_DIR)
    except Exception as e:
        logger.critical(f"Pipeline failed due to unexpected error: {str(e)}")
