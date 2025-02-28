"""
test_pipeline.py
----------------
Tests for the pipeline functions defined in pipeline.py.
"""

import os
import numpy as np
import pandas as pd
import pytest

# Adjust sys.path to import from the src directory
import sys
from os.path import abspath, join, dirname
sys.path.insert(0, abspath(join(dirname(__file__), '..', 'src')))

from pipeline import (
    segment_text,
    generate_embeddings,
    classify_local,
    cluster_sentences,
    classify_global,
    merge_local_global,
    resolve_conflict,
    final_classification
)

# Create a dummy configuration dictionary for testing purposes.
dummy_config = {
    "preprocessing": {
        "context_window": 1
    },
    "embedding": {
        "model_name": "all-MiniLM-L6-v2"
    },
    "clustering": {
        "hdbscan": {
            "min_cluster_size": 2,
            "metric": "euclidean"
        }
    },
    "classification": {
        "local": {
            "prompt": "Dummy prompt for local classification",
            "confidence_threshold": 0.8
        },
        "global": {
            "prompt": "Dummy prompt for global classification"
        },
        "final": {
            "final_weight_local": 0.6,
            "final_weight_global": 0.4
        }
    },
    "paths": {
        "data_dir": "data/interviews",
        "logs_dir": "logs",
        "output_dir": "output"
    }
}


############################################
# Test for segment_text
############################################
def test_segment_text():
    sample_text = "Sentence one. Sentence two? Sentence three!"
    sentences = segment_text(sample_text)
    
    # Expect three sentences
    assert isinstance(sentences, list)
    assert len(sentences) == 3
    
    # Check that each sentence has an 'id' and 'sentence' key
    for idx, sent_dict in enumerate(sentences):
        assert "id" in sent_dict
        assert "sentence" in sent_dict
        # Ensure that IDs are sequential starting at 0
        assert sent_dict["id"] == idx
        # Ensure the sentence text is non-empty
        assert isinstance(sent_dict["sentence"], str) and len(sent_dict["sentence"]) > 0


############################################
# Test for generate_embeddings
############################################
def test_generate_embeddings():
    # Create dummy sentences as would be returned by segment_text
    sentences = [
        {"id": 0, "sentence": "This is the first sentence."},
        {"id": 1, "sentence": "This is the second sentence."},
        {"id": 2, "sentence": "This is the third sentence."}
    ]
    # Generate embeddings with a context window of 1
    embeddings = generate_embeddings(sentences, context_window=1)
    
    # Check that embeddings is a numpy array with the same number of rows as sentences.
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(sentences)
    
    # Optionally, check that the embedding dimension is positive.
    assert embeddings.shape[1] > 0


############################################
# Test for classify_local
############################################
def test_classify_local():
    # Create dummy sentences
    sentences = [
        {"id": 0, "sentence": "This is the first sentence."},
        {"id": 1, "sentence": "This is the second sentence."}
    ]
    # Generate dummy embeddings (the actual values won't affect dummy classification)
    embeddings = np.random.rand(len(sentences), 384)
    
    df_local = classify_local(sentences, embeddings, dummy_config)
    
    # Check that the DataFrame has the required columns
    for col in ["id", "sentence", "local_label", "local_confidence"]:
        assert col in df_local.columns
    
    # Check that confidence scores are within expected range (0.7 to 1.0)
    confidences = df_local["local_confidence"].values
    assert np.all(confidences >= 0.7) and np.all(confidences <= 1.0)


############################################
# Test for cluster_sentences
############################################
def test_cluster_sentences():
    # Create dummy embeddings for 5 sentences
    dummy_embeddings = np.random.rand(5, 384)
    cluster_labels = cluster_sentences(dummy_embeddings, dummy_config)
    
    # Check that cluster_labels is a numpy array with length equal to number of sentences
    assert isinstance(cluster_labels, np.ndarray)
    assert len(cluster_labels) == dummy_embeddings.shape[0]


############################################
# Test for classify_global
############################################
def test_classify_global():
    # Create dummy sentences
    sentences = [
        {"id": 0, "sentence": "Sentence one."},
        {"id": 1, "sentence": "Sentence two."},
        {"id": 2, "sentence": "Sentence three."}
    ]
    # Create dummy embeddings
    dummy_embeddings = np.random.rand(len(sentences), 384)
    # Create dummy cluster labels (simulate two clusters: 0 and -1 for outlier)
    dummy_cluster_labels = np.array([0, 0, -1])
    
    df_global = classify_global(sentences, dummy_embeddings, dummy_cluster_labels, dummy_config)
    
    # Check that the returned DataFrame has the required columns
    for col in ["id", "global_label", "cluster"]:
        assert col in df_global.columns
    
    # Check that the global_label for outlier is "Unassigned"
    outlier_label = df_global[df_global["cluster"] == -1]["global_label"].iloc[0]
    assert outlier_label == "Unassigned"


############################################
# Test for resolve_conflict
############################################
def test_resolve_conflict():
    # Case 1: High local confidence should choose local label.
    local_label = "Local_Topic"
    global_label = "Global_Topic"
    local_confidence = 0.9  # above threshold 0.8
    result = resolve_conflict(local_label, global_label, local_confidence, dummy_config)
    assert result == local_label
    
    # Case 2: Low local confidence should choose global label.
    local_confidence = 0.7  # below threshold
    result = resolve_conflict(local_label, global_label, local_confidence, dummy_config)
    assert result == global_label


############################################
# Test for merge_local_global
############################################
def test_merge_local_global():
    # Create dummy local DataFrame
    df_local = pd.DataFrame({
        "id": [0, 1],
        "sentence": ["Sentence one.", "Sentence two."],
        "local_label": ["Local_A", "Local_B"],
        "local_confidence": [0.85, 0.75]
    })
    # Create dummy global DataFrame
    df_global = pd.DataFrame({
        "id": [0, 1],
        "global_label": ["Global_A", "Global_B"],
        "cluster": [0, 0]
    })
    merged_df = merge_local_global(df_local, df_global, dummy_config)
    
    # Check that merged_df has a final_label column
    assert "final_label" in merged_df.columns
    # For id 0: confidence 0.85 (>=0.8) so expect local label "Local_A"
    final_label_0 = merged_df[merged_df["id"] == 0]["final_label"].iloc[0]
    assert final_label_0 == "Local_A"
    # For id 1: confidence 0.75 (<0.8) so expect global label "Global_B"
    final_label_1 = merged_df[merged_df["id"] == 1]["final_label"].iloc[0]
    assert final_label_1 == "Global_B"


############################################
# Test for final_classification
############################################
def test_final_classification():
    # Create a dummy merged DataFrame
    df = pd.DataFrame({
        "id": [0, 1],
        "sentence": ["Sentence one.", "Sentence two."],
        "local_label": ["Local_A", "Local_B"],
        "local_confidence": [0.85, 0.75],
        "global_label": ["Global_A", "Global_B"],
        "cluster": [0, 0],
        "final_label": ["Local_A", "Global_B"]
    })
    
    final_df = final_classification(df, dummy_config)
    
    # Check that final_df has a new column "local_conf_norm"
    assert "local_conf_norm" in final_df.columns
    # Check that local_conf_norm is a numeric column with the same number of rows as df
    assert final_df["local_conf_norm"].dtype.kind in "fc"
    assert len(final_df) == len(df)
