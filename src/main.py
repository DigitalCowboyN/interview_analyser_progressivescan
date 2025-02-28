#!/usr/bin/env python3
if __name__ == "__main__" and __package__ is None:
    import os
    import sys
    # Add the parent directory to sys.path so that 'src' is recognized as a package.
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    __package__ = "src"

import os
import glob
import argparse
import logging
from .utils import setup_logger, create_dir, save_json, load_yaml
from .pipeline import (
    segment_text,
    generate_embeddings,
    classify_local,
    cluster_sentences,
    classify_global,
    merge_local_global,
    final_classification
)

def main():
    # Initialize logger
    logger = setup_logger(log_file="logs/pipeline.log")
    logger.info("Starting Interview Analyzer Ensemble Pipeline")

    # Load configuration from config.yaml
    config_path = "config.yaml"
    config = load_yaml(config_path)
    logger.info(f"Loaded configuration from {config_path}")

    # Ensure necessary directories exist
    create_dir(config["paths"]["logs_dir"])
    create_dir(config["paths"]["output_dir"])

    # Use glob to find all .txt files in the data directory
    data_dir = config["paths"]["data_dir"]
    txt_files = glob.glob(os.path.join(data_dir, "*.txt"))

    if not txt_files:
        logger.warning(f"No .txt files found in {data_dir}.")
        return

    for input_file in txt_files:
        logger.info(f"Processing input file: {input_file}")
        
        # Read the transcript file
        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read()

        # Step 1: Sentence segmentation
        sentences = segment_text(text)
        logger.info(f"Segmented text into {len(sentences)} sentences.")

        # Step 2: Generate embeddings with context
        embeddings = generate_embeddings(sentences, context_window=config["preprocessing"]["context_window"])
        logger.info("Generated embeddings for all sentences.")

        # Step 3: Local Classification using LLaMA-2 (via prompt engineering)
        df_local = classify_local(sentences, embeddings, config)
        logger.info("Completed local classification.")

        # Step 4: Global Thematic Clustering and Classification
        cluster_labels = cluster_sentences(embeddings, config)
        df_global = classify_global(sentences, embeddings, cluster_labels, config)
        logger.info("Completed global classification.")

        # Step 5: Merge local and global outputs with conflict resolution
        merged_df = merge_local_global(df_local, df_global, config)
        logger.info("Merged local and global outputs.")

        # Step 6: Final Meta-Classification
        final_df = final_classification(merged_df, config)
        logger.info("Final classification complete.")

        # Create an output filename based on the input filename
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = f"{base_name}_final_output.json"
        output_path = os.path.join(config["paths"]["output_dir"], output_file)

        # Save final output as JSON
        final_output = final_df.to_dict(orient="records")
        save_json(final_output, output_path)
        logger.info(f"Final output saved to {output_path}")

if __name__ == "__main__":
    main()
