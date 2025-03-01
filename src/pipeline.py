"""
pipeline.py
-----------
This module contains the main pipeline functions for processing interview transcripts.
It covers sentence segmentation, embedding generation with context, local classification,
thematic clustering and global classification, merging, conflict resolution, and final classification.
"""

import re
import numpy as np
import pandas as pd
import spacy
import logging
import yaml
from sentence_transformers import SentenceTransformer
import hdbscan
from sklearn.preprocessing import StandardScaler

# Load configuration from config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load spaCy model (English)
nlp = spacy.load("en_core_web_sm")

# Global logging instance (assumes logging is already configured)
logger = logging.getLogger(__name__)

# Initialize Sentence Transformer model for embedding generation.
embedder = SentenceTransformer(config["embedding"]["model_name"])

# Import tqdm for progress bars.
from tqdm import tqdm

# Import the LLaMA-2 model loader from ctransformers.
try:
    from ctransformers import AutoModelForCausalLM
except ImportError:
    raise ImportError("ctransformers library not installed. Please install it to use LLaMA-2.")

############################################
# 1. Sentence Segmentation
############################################
def segment_text(text):
    """
    Splits unstructured text into sentences using spaCy.
    
    Args:
        text (str): The full unstructured transcript text.
    
    Returns:
        list of dict: Each dict contains 'id' and 'sentence' keys.
    """
    logger.info("Segmenting text into sentences using spaCy.")
    doc = nlp(text)
    sentences = []
    for i, sent in enumerate(doc.sents):
        sentence_text = sent.text.strip()
        if sentence_text:
            sentences.append({"id": i, "sentence": sentence_text})
    logger.info(f"Segmented text into {len(sentences)} sentences.")
    return sentences

############################################
# 2. Embedding Generation with Contextual Enrichment
############################################
from concurrent.futures import ThreadPoolExecutor

def generate_embeddings(sentences, context_window=1):
    """
    Generate embeddings for each sentence and enrich them with a context window.
    
    Args:
        sentences (list of dict): List with each sentence and its ID.
        context_window (int): Number of neighboring sentences to include on each side.
    
    Returns:
        np.array: Array of enriched embeddings.
    """
    logger.info("Generating base embeddings using Sentence Transformers.")
    
    if not sentences:
        logger.warning("No sentences provided for embedding generation.")
        return np.zeros((1, config["embedding"]["embedding_dim"]))

    sentence_texts = [s["sentence"] for s in sentences if s["sentence"].strip()]

    if not sentence_texts:
        logger.warning("All sentences are empty after preprocessing.")
        return np.zeros((1, config["embedding"]["embedding_dim"]))

    try:
        base_embeddings = embedder.encode(sentence_texts, convert_to_numpy=True)
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return np.zeros((1, config["embedding"]["embedding_dim"]))

    def compute_context_embedding(i):
        start = max(0, i - context_window)
        end = min(len(base_embeddings), i + context_window + 1)
        return np.mean(base_embeddings[start:end], axis=0)

    with ThreadPoolExecutor(max_workers=2) as executor:
        enriched_embeddings = list(executor.map(compute_context_embedding, range(len(base_embeddings))))

    logger.info("Generated context-aware embeddings for all sentences.")
    return np.array(enriched_embeddings)

############################################
# Initialize the LLaMA-2 model using the path from config.yaml
############################################
llama_model_path = config["classification"]["local"].get("llama_model_path", "/models/llama-2-7b-chat.Q4_K_M.gguf")
llama_model = AutoModelForCausalLM.from_pretrained(llama_model_path, gpu_layers=0)
llama_model.max_new_tokens = 50  # Increase response length to improve completeness

############################################
# 3. Local Classification (Sentence-Level)
############################################
def aggregate_local_context(idx, sentences, embeddings, config):
    """
    Generate a context string for a given sentence while ensuring it fits within the model's token limit.
    
    Args:
        idx (int): The index of the current sentence.
        sentences (list of dict): The full list of sentence dictionaries.
        embeddings (np.array): Array of enriched embeddings.
        config (dict): Configuration parameters that include context aggregation settings.
    
    Returns:
        str: A truncated context string optimized for token limits.
    """
    max_tokens = 512  # Model's token limit
    estimated_token_ratio = 1.3  # Approximate words-to-tokens ratio
    max_context_tokens = max_tokens // 2  # Reserve space for prompt & response

    # Define context window sizes for multiple passes
    context_windows = [1, 2, 4, 7]
    selected_context = []

    for window in context_windows:
        start = max(0, idx - window)
        end = min(len(sentences), idx + window + 1)
        context_sentences = [sentences[i]["sentence"] for i in range(start, end) if i != idx]

        # Estimate token count
        estimated_tokens = sum(len(sent.split()) for sent in context_sentences) / estimated_token_ratio

        if estimated_tokens <= max_context_tokens:
            selected_context = context_sentences
        else:
            break  # Stop expanding if we exceed the token limit

    # Join selected context sentences
    context_str = " ".join(selected_context)

    return context_str

from concurrent.futures import ThreadPoolExecutor

def classify_local(sentences, embeddings, config):
    """
    Classify each sentence individually using LLaMA-2 via prompt engineering.
    This function builds two prompts from the configured templates (one without context
    and one with context using an aggregated context), sends them to the model, and parses
    the responses to extract the topic labels and confidence scores.
    
    Args:
        sentences (list of dict): List of sentence dictionaries with keys 'id' and 'sentence'.
        embeddings (np.array): Array of enriched embeddings from generate_embeddings.
        config (dict): Configuration parameters containing the local prompts, thresholds, and context aggregation settings.
    
    Returns:
        pd.DataFrame: DataFrame with columns: id, sentence, local_label_no_context,
                      local_confidence_no_context, local_label_with_context, local_confidence_with_context.
    """
    logger.info("Performing local classification using LLaMA-2 with two separate prompts (with and without context).")
    results = []
    # Retrieve separate prompt templates and the confidence threshold from config.
    prompt_no_context = config["classification"]["local"]["prompt_no_context"]
    prompt_with_context = config["classification"]["local"]["prompt_with_context"]
    confidence_threshold = config["classification"]["local"]["confidence_threshold"]

    def classify_sentence(idx, item, prompt_no_context, prompt_with_context, confidence_threshold, sentences, embeddings):
        """
        Classify a single sentence using LLaMA-2 with and without context.
        """
        try:
            full_prompt_no_context = (
                f"{prompt_no_context}\n\n"
                f"Input Sentence: \"{item['sentence']}\"\n"
                f"Context: []\n\n"
                f"Please provide a single-word category that best describes the sentence.\n"
                f"Example: Marketing, Sales, CustomerSupport, Finance.\n"
                f"Do not provide any additional explanation."
            )
            try:
                response_no_context = llama_model(full_prompt_no_context, max_new_tokens=30).strip()
                logger.debug(f"Raw model response (no context) for sentence ID {item['id']}: {response_no_context}")
                if not response_no_context or "<" not in response_no_context or "[" not in response_no_context:
                    raise ValueError("Malformed response from model.")
                logger.debug(f"Model response (no context) for sentence ID {item['id']}: {response_no_context}")
            except Exception as e:
                logger.error(f"Error generating response (no context) for sentence ID {item['id']}: {e}")
                response_no_context = "<Unknown> [0.0]"
        except Exception as e:
            logger.error(f"Error during local classification for sentence ID {item['id']}: {e}")
            return None

        label_match_no_context = re.search(r"\b([A-Za-z]+)\b", response_no_context)
        confidence_match_no_context = re.search(r"\b(0\.\d+|1\.0)\b", response_no_context)

        if label_match_no_context and confidence_match_no_context:
            logger.debug(f"Extracted label: {label_match_no_context.group(1)}, confidence: {confidence_match_no_context.group(1)}")
            label_no_context = label_match_no_context.group(1).strip()
            try:
                confidence_no_context = float(confidence_match_no_context.group(1).strip())
            except ValueError:
                confidence_no_context = 0.0
        else:
            logger.warning(f"Malformed response (no context) for sentence ID {item['id']}: {response_no_context}")
            label_no_context = "Unknown"
            confidence_no_context = 0.0
        try:
            confidence_no_context = float(confidence_match_no_context.group(1).strip()) if confidence_match_no_context else 0.0
        except ValueError:
            confidence_no_context = 0.0

        if confidence_no_context < confidence_threshold:
            label_no_context = "Unknown"

        # Perform multiple passes with increasing context window sizes
        context_windows = [1, 2, 4, 7]
        best_label = "Unknown"
        best_confidence = 0.0

        for window in context_windows:
            start = max(0, idx - window)
            end = min(len(sentences), idx + window + 1)
            context_sentences = [sentences[i]["sentence"] for i in range(start, end) if i != idx]
            context_str = " ".join(context_sentences)

            full_prompt_with_context = (
                f"{prompt_with_context}\n\n"
                f"Input Sentence: \"{item['sentence']}\"\n"
                f"Context: {context_str}"
            )

            response_with_context = llama_model(full_prompt_with_context, max_new_tokens=30)
            logger.debug(f"Model response (context window {window}) for sentence ID {item['id']}: {response_with_context}")

            label_match_with_context = re.search(r"<(.*?)>", response_with_context)
            confidence_match_with_context = re.search(r"\[(.*?)\]", response_with_context)

            label_with_context = label_match_with_context.group(1).strip() if label_match_with_context else "Unknown"
            try:
                confidence_with_context = float(confidence_match_with_context.group(1).strip()) if confidence_match_with_context else 0.0
            except ValueError:
                confidence_with_context = 0.0

            if confidence_with_context > best_confidence:
                best_label = label_with_context
                best_confidence = confidence_with_context
        full_prompt_with_context = (
            f"{prompt_with_context}\n\n"
            f"Input Sentence: \"{item['sentence']}\"\n"
            f"Context: {context_str}"
        )
        response_with_context = llama_model(full_prompt_with_context, max_new_tokens=30)
        logger.debug(f"Model response (with context) for sentence ID {item['id']}: {response_with_context}")

        label_match_with_context = re.search(r"<(.*?)>", response_with_context)
        confidence_match_with_context = re.search(r"\[(.*?)\]", response_with_context)

        label_with_context = label_match_with_context.group(1).strip() if label_match_with_context else "Unknown"
        try:
            confidence_with_context = float(confidence_match_with_context.group(1).strip()) if confidence_match_with_context else 0.0
        except ValueError:
            confidence_with_context = 0.0

        if confidence_with_context < confidence_threshold:
            label_with_context = "Unknown"

        return {
            "id": item["id"],
            "sentence": item["sentence"],
            "local_label_no_context": label_no_context,
            "local_confidence_no_context": confidence_no_context,
            "local_label_with_context": label_with_context,
            "local_confidence_with_context": confidence_with_context
        }

    with ThreadPoolExecutor(max_workers=1) as executor:
        results = list(executor.map(classify_sentence, range(len(sentences)), sentences))

    # Wrap the loop in a tqdm progress bar.
from concurrent.futures import ThreadPoolExecutor

def classify_sentence(idx, item, prompt_no_context, prompt_with_context, confidence_threshold, sentences, embeddings):
        try:
            full_prompt_no_context = (
                f"{prompt_no_context}\n\n"
                f"Input Sentence: \"{item['sentence']}\"\n"
                f"Context: []"
            )
            import subprocess
            import shlex

            safe_prompt = full_prompt_no_context.replace('"', '\\"').replace("'", "\\'").replace("\n", " ")
            command = f'python -c "from ctransformers import AutoModelForCausalLM; model = AutoModelForCausalLM.from_pretrained(\'{llama_model_path}\', gpu_layers=0); print(model(\'{safe_prompt}\', max_new_tokens=30))"'
            try:
                response_no_context = subprocess.check_output(shlex.split(command), text=True, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                logger.error(f"Subprocess error: {e.output}")
                response_no_context = "<Unknown> [0.0]"
            logger.debug(f"Model response (no context) for sentence ID {item['id']}: {response_no_context}")
        except Exception as e:
            logger.error(f"Error during local classification for sentence ID {item['id']}: {e}")
            return None

        # Extract label (in angle brackets) and confidence (in square brackets)
        label_match_no_context = re.search(r"<(.*?)>", response_no_context)
        confidence_match_no_context = re.search(r"\[(.*?)\]", response_no_context)
        
        label_no_context = label_match_no_context.group(1).strip() if label_match_no_context else "Unknown"
        confidence_no_context = 0.8  # Assign a default confidence score
        
        if confidence_no_context < confidence_threshold:
            label_no_context = "Unknown"

        # ----- Classification with context -----
        # Retrieve aggregated context based on the config setting.
        context_str = aggregate_local_context(idx, sentences, embeddings, config)
        full_prompt_with_context = (
            f"{prompt_with_context}\n\n"
            f"Input Sentence: \"{item['sentence']}\"\n"
            f"Context: {context_str}"
        )
        response_with_context = llama_model(full_prompt_with_context, max_new_tokens=30)
        logger.debug(f"Model response (with context) for sentence ID {item['id']}: {response_with_context}")

        label_match_with_context = re.search(r"<(.*?)>", response_with_context)
        confidence_match_with_context = re.search(r"\[(.*?)\]", response_with_context)
        
        label_with_context = label_match_with_context.group(1).strip() if label_match_with_context else "Unknown"
        try:
            confidence_with_context = float(confidence_match_with_context.group(1).strip()) if confidence_match_with_context else 0.0
        except ValueError:
            confidence_with_context = 0.0
        
        if confidence_with_context < confidence_threshold:
            label_with_context = "Unknown"


def classify_local(sentences, embeddings, config):
    logger.info("Performing local classification using LLaMA-2 with two separate prompts (with and without context).")
    results = []
    prompt_no_context = config["classification"]["local"]["prompt_no_context"]
    prompt_with_context = config["classification"]["local"]["prompt_with_context"]
    confidence_threshold = config["classification"]["local"]["confidence_threshold"]

    def classify_sentence_wrapper(idx, item):
        return classify_sentence(idx, item, prompt_no_context, prompt_with_context, confidence_threshold, sentences, embeddings)

    with ThreadPoolExecutor(max_workers=1) as executor:
        results = list(executor.map(lambda idx_item: classify_sentence_wrapper(*idx_item), enumerate(sentences)))

    failed_count = sum(1 for r in results if r is None)
    logger.warning(f"Failed classifications: {failed_count}")
    results = [r for r in results if r is not None]  # Remove None values
    logger.debug(f"Valid local classification results: {len(results)}")
    if not results:
        logger.error("No valid classification results. Returning empty DataFrame.")
        return pd.DataFrame(columns=["id", "sentence", "local_label_no_context", "local_confidence_no_context", "local_label_with_context", "local_confidence_with_context"])

    df_local = pd.DataFrame(results).dropna(subset=["id"])
    logger.info("Local classification completed for all sentences.")
    return df_local


############################################
# 4. Global Thematic Clustering & Classification
############################################
def cluster_sentences(embeddings, config):
    """
    Cluster sentence embeddings using HDBSCAN.
    
    Args:
        embeddings (np.array): Array of enriched sentence embeddings.
        config (dict): Configuration parameters for clustering.
    
    Returns:
        np.array: Array of cluster labels (with -1 for outliers).
    """
    logger.info("Clustering sentence embeddings using HDBSCAN.")
    hdbscan_params = config["clustering"]["hdbscan"]
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=hdbscan_params.get("min_cluster_size", 5),
        metric=hdbscan_params.get("metric", "euclidean")
    )
    cluster_labels = clusterer.fit_predict(embeddings)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    logger.info(f"HDBSCAN produced {n_clusters} clusters (excluding outliers).")
    return cluster_labels

def aggregate_global_context(cluster_sentences, cluster_embeddings, config):
    """
    Generate a context string for a cluster while ensuring it fits within the model's token limit.
    
    Args:
        cluster_sentences (list of str): All sentences in the cluster.
        cluster_embeddings (np.array): Embeddings for the sentences in the cluster.
        config (dict): Configuration parameters for global context aggregation.
        
    Returns:
        tuple: A tuple containing:
            - cluster_summary (str): A brief summary or representative text from the cluster.
            - context_str (str): A truncated context string optimized for token limits.
    """
    max_tokens = 512  # Model's token limit
    estimated_token_ratio = 1.3  # Approximate words-to-tokens ratio
    max_context_tokens = max_tokens // 2  # Reserve space for prompt & response

    method = config["classification"]["global"].get("context_aggregation_method", "representative_sentences")
    summary_count = config["classification"]["global"].get("summary_sentence_count", 3)

    # Create a brief summary using the most relevant sentences
    cluster_summary = " ".join(cluster_sentences[:summary_count])

    if method == "representative_sentences":
        selected_context = cluster_sentences[:summary_count]
    elif method == "summary":
        selected_context = [f"Summary: {cluster_summary}"]
    elif method == "embedding":
        if len(cluster_embeddings) > 0:
            avg_embedding = cluster_embeddings.mean(axis=0)
            selected_context = [", ".join([f"{x:.4f}" for x in avg_embedding])]
        else:
            selected_context = ["[]"]
    else:
        selected_context = []

    # Estimate token count and truncate if necessary
    estimated_tokens = sum(len(sent.split()) for sent in selected_context) / estimated_token_ratio
    while estimated_tokens > max_context_tokens and len(selected_context) > 1:
        selected_context.pop()  # Remove the last sentence to reduce token count
        estimated_tokens = sum(len(sent.split()) for sent in selected_context) / estimated_token_ratio

    context_str = " ".join(selected_context)

    return cluster_summary, context_str

from tqdm import tqdm

def classify_global(sentences, embeddings, cluster_labels, config):
    """
    Perform global thematic classification on clusters. For each cluster (except outliers),
    this function aggregates a cluster summary and context (using an abstracted method from config),
    builds a prompt using the global prompt template, and sends it to LLaMA-2. The response is parsed
    to extract a thematic label and confidence score. If the confidence is below the configured threshold,
    the label is set to "Unknown".
    
    Args:
        sentences (list of dict): List of sentence dictionaries.
        embeddings (np.array): Array of enriched embeddings.
        cluster_labels (np.array): Cluster labels for each sentence.
        config (dict): Configuration with global classification parameters.
    
    Returns:
        pd.DataFrame: DataFrame with columns: id, global_label, global_confidence, cluster.
    """
    logger.info("Performing global thematic classification on clusters.")
    df = pd.DataFrame({
        "id": [s["id"] for s in sentences],
        "sentence": [s["sentence"] for s in sentences],
        "cluster": cluster_labels
    })
    
    global_results = []
    global_prompt = config["classification"]["global"]["prompt"]
    global_conf_threshold = config["classification"]["global"].get("confidence_threshold", 0.6)
    
    unique_clusters = sorted(df["cluster"].unique())
    for cluster in tqdm(unique_clusters, desc="Global Classification"):
        if cluster == -1:
            # Handle outliers by assigning them to the closest cluster based on embedding similarity.
            logger.warning(f"Handling outlier cluster {cluster}. Assigning to nearest cluster.")
            outlier_indices = df[df["cluster"] == cluster].index.tolist()
            outlier_embeddings = embeddings[outlier_indices]

            # Compute similarity to all other clusters
            cluster_means = {c: embeddings[df[df["cluster"] == c].index].mean(axis=0) for c in unique_clusters if c != -1}
            if not cluster_means:
                logger.warning("No valid clusters found. Skipping outlier reassignment.")
                continue  # Skip reassignment if no valid clusters exist

            closest_cluster = min(cluster_means, key=lambda c: np.linalg.norm(outlier_embeddings.mean(axis=0) - cluster_means[c]))

            # Assign outliers to the closest cluster
            df.loc[outlier_indices, "cluster"] = closest_cluster
            cluster = closest_cluster  # Continue processing as a normal cluster
        else:
            # Get all sentences for the cluster.
            cluster_sentence_list = df[df["cluster"] == cluster]["sentence"].tolist()
            # Get corresponding embeddings.
            cluster_indices = [i for i, cl in enumerate(cluster_labels) if cl == cluster]
            cluster_embs = embeddings[cluster_indices] if len(cluster_indices) > 0 else None
            
            # Use helper function to aggregate context.
            cluster_summary, context_str = aggregate_global_context(cluster_sentence_list, cluster_embs, config)
            
            # Build the prompt including both the cluster summary and the aggregated context.
            full_prompt = (
                f"{global_prompt}\n\n"
                f"Cluster Summary: \"{cluster_summary}\"\n"
                f"Context: {context_str}"
            )
            
            response = llama_model(full_prompt, max_new_tokens=30)
            logger.debug(f"Model response for cluster {cluster}: {response}")
            
            # Expect the response to include a label in angle brackets and a confidence in square brackets.
            label_match = re.search(r"<(.*?)>", response)
            confidence_match = re.search(r"\[(.*?)\]", response)
            
            global_label = label_match.group(1).strip() if label_match else "Unknown"
            try:
                confidence = float(confidence_match.group(1).strip()) if confidence_match else 0.0
            except ValueError:
                confidence = 0.0
            
            global_conf = confidence
            if global_conf < global_conf_threshold:
                global_label = "Unknown"
            
            cluster_ids = df[df["cluster"] == cluster]["id"].tolist()
        
        global_results.append({
            "cluster": cluster,
            "global_label": global_label,
            "global_confidence": global_conf,
            "sentence_ids": cluster_ids
        })
    
    # Map the global label and confidence for each sentence based on its cluster.
    global_label_map = {}
    global_conf_map = {}
    for entry in global_results:
        for sid in entry["sentence_ids"]:
            global_label_map[sid] = entry["global_label"]
            global_conf_map[sid] = entry["global_confidence"]
    
    df["global_label"] = df["id"].apply(lambda x: global_label_map.get(x, "Unassigned"))
    df["global_confidence"] = df["id"].apply(lambda x: global_conf_map.get(x, 1.0))
    logger.info("Global thematic classification completed.")
    return df[["id", "global_label", "global_confidence", "cluster"]]


############################################
# 5. Merging Local & Global Outputs and Conflict Resolution
############################################
def merge_local_global(df_local, df_global, config):
    """
    Merge local and global classification results on sentence ID.
    
    Args:
        df_local (pd.DataFrame): DataFrame with local classification results.
        df_global (pd.DataFrame): DataFrame with global thematic classification.
        config (dict): Configuration for conflict resolution.
    
    Returns:
        pd.DataFrame: Merged DataFrame including a preliminary final label.
    """
    logger.info("Merging local and global outputs.")
    logger.debug(f"df_local columns: {df_local.columns}")
    logger.debug(f"df_global columns: {df_global.columns}")
    
    merged_df = pd.merge(df_local, df_global, on="id", how="left")
    merged_df["final_label"] = merged_df.apply(
        lambda row: resolve_conflict(
            row["local_label_with_context"],
            row["global_label"],
            row["local_confidence_with_context"],
            row["global_confidence"],
            config
        ),
        axis=1
    )
    logger.info("Merging and conflict resolution completed.")
    return merged_df

def resolve_conflict(local_label, global_label, local_confidence, global_confidence, config):
    """
    Resolve conflicts between local and global labels using weighted rules.
    
    The function computes a weighted score for both local and global predictions
    based on their confidence and configured weights. If a classifier's confidence is
    below its threshold, its contribution is set to zero.
    
    Args:
        local_label (str): Label predicted at the local level.
        global_label (str): Label predicted at the global (cluster) level.
        local_confidence (float): Confidence score from local classification.
        global_confidence (float): Confidence score from global classification.
        config (dict): Configuration parameters including thresholds and weights.
    
    Returns:
        str: The final chosen label.
    """
    local_threshold = config["classification"]["local"].get("confidence_threshold", 0.8)
    global_threshold = config["classification"]["global"].get("confidence_threshold", 0.6)
    weight_local = config["classification"]["final"].get("final_weight_local", 0.6)
    weight_global = config["classification"]["final"].get("final_weight_global", 0.4)
    
    if local_confidence >= local_threshold and global_confidence >= global_threshold:
        weight_local = local_confidence / (local_confidence + global_confidence)
        weight_global = global_confidence / (local_confidence + global_confidence)
    else:
        weight_local = config["classification"]["final"]["final_weight_local"]
        weight_global = config["classification"]["final"]["final_weight_global"]

    local_weighted = weight_local * local_confidence if local_confidence >= local_threshold else 0.0
    global_weighted = weight_global * global_confidence if global_confidence >= global_threshold else 0.0
    
    if local_weighted >= global_weighted and local_weighted > 0:
        return local_label
    elif global_weighted > local_weighted:
        return global_label
    else:
        return "Unknown"


############################################
# 6. Final Meta-Classification / Post-Processing
############################################
def final_classification(merged_df, config):
    """
    Perform final meta-classification. This function is a placeholder for
    additional feature engineering or meta-classifier training.
    
    Args:
        merged_df (pd.DataFrame): DataFrame after merging local and global outputs.
        config (dict): Configuration parameters.
    
    Returns:
        pd.DataFrame: Final DataFrame with classification results.
    """
    logger.info("Performing final meta-classification.")
    from sklearn.preprocessing import StandardScaler
    if merged_df.empty:
        logger.error("Merged DataFrame is empty. Skipping final classification.")
        return merged_df

    scaler = StandardScaler()
    merged_df["local_conf_norm"] = scaler.fit_transform(merged_df[["local_confidence_with_context"]])
    logger.info("Final classification complete.")
    return merged_df
