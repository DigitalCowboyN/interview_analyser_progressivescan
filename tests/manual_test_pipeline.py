import sys
import os

# Ensure the parent directory is in sys.path so 'src' is recognized as a package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import faulthandler

faulthandler.enable()
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
from src.pipeline import (
    segment_text,
    generate_embeddings,
    classify_local,
    cluster_sentences,
    classify_global,
    merge_local_global,
    final_classification
)

# Load config.yaml for test settings
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def run_manual_test():
    """Run the pipeline on a short text snippet for quick testing."""
    
    # Sample unstructured text (simulating a small interview snippet)
    sample_text = """
    I think our marketing strategy needs improvement. 
    We should focus more on digital campaigns. 
    Customer engagement has been low recently.
    """

    print("Running manual test with sample text...")

    # Step 1: Sentence Segmentation
    sentences = segment_text(sample_text)
    print(f"Segmented into {len(sentences)} sentences.")

    # Step 2: Generate Embeddings
    embeddings = generate_embeddings(sentences, context_window=config["preprocessing"]["context_window"])
    print("Generated embeddings.")

    print("Starting local classification...")
    logger.debug(f"Sentences: {sentences}")
    logger.debug(f"Embeddings shape: {embeddings.shape if hasattr(embeddings, 'shape') else type(embeddings)}")
    logger.debug(f"Config parameters: {config.get('local_classification', {})}")
    logger.debug(f"Embeddings shape: {embeddings.shape if hasattr(embeddings, 'shape') else type(embeddings)}")

    # Step 3: Local Classification
    try:
        logger.info("Calling classify_local() with %d sentences and embeddings of shape %s", len(sentences), embeddings.shape)
        df_local = classify_local(sentences, embeddings, config)
        logger.info("classify_local() completed successfully.")
        logger.debug(f"Local classification results: {df_local}")
        logger.debug(f"Local classification results: {df_local}")
        logger.debug(f"Local classification results: {df_local}")
    except Exception as e:
        logger.error(f"Error during local classification: {e}", exc_info=True)
        return
    print("Completed local classification.")
    print(df_local)  # Debugging: Print local classification results

    # Step 4: Clustering
    cluster_labels = cluster_sentences(embeddings, config)
    print("Performed clustering.")
    print(cluster_labels)  # Debugging: Print cluster labels

    # Step 5: Global Classification
    df_global = classify_global(sentences, embeddings, cluster_labels, config)
    print("Completed global classification.")
    print(df_global)  # Debugging: Print global classification results

    # Step 6: Merge Local & Global
    merged_df = merge_local_global(df_local, df_global, config)
    print("Merged local and global classifications.")

    # Step 7: Final Classification
    final_df = final_classification(merged_df, config)
    print("Final classification complete.")

    print("\nFinal Output:")
    print(final_df)

if __name__ == "__main__":
    run_manual_test()
