# Interview Analyzer

## Overview
The Interview Analyzer is a pipeline designed to process interview transcripts, segmenting text, generating embeddings, classifying sentences, and clustering themes. It utilizes advanced natural language processing techniques, including the Sentence Transformers library and the LLaMA-2 model for classification tasks.

## Features
- **Text Segmentation**: Splits unstructured interview text into individual sentences.
- **Embedding Generation**: Generates contextual embeddings for sentences using the Sentence Transformers library.
- **Local and Global Classification**: Classifies sentences at both local and global levels, providing thematic labels and confidence scores.
- **Clustering**: Groups similar sentences into clusters using HDBSCAN.
- **Conflict Resolution**: Merges local and global classifications, resolving conflicts based on confidence scores.

## Project Structure
```
.
├── .gitignore
├── Dockerfile
├── README.md
├── config.yaml
├── requirements.txt
├── src
│   ├── __init__.py
│   ├── config.py
│   ├── pipeline
│   │   ├── __init__.py
│   │   └── pipeline.py
│   └── utils.py
└── tests
    ├── test_pipeline.py
    └── manual_test_pipeline.py
```

## Installation

### Prerequisites
- Python 3.10 or higher
- Docker (for containerized deployment)

### Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd interview_analyzer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Build and run using Docker:
   ```bash
   docker build -t interview_analyzer .
   docker run -it interview_analyzer
   ```

## Configuration
The project uses a `config.yaml` file to manage settings. You can customize parameters such as:
- **Embedding Model**: Specify the model name for sentence embeddings.
- **Clustering Parameters**: Adjust HDBSCAN settings for clustering.
- **Classification Prompts**: Modify prompts used for local and global classification.

## Usage
To run the pipeline, execute the following command:
```bash
python src/main.py
```

### Manual Testing
For manual testing, you can run:
```bash
python tests/manual_test_pipeline.py
```

## Testing
The project includes unit tests located in the `tests` directory. To run the tests, use:
```bash
pytest tests/
```

## Logging
Logs are generated in the `logs` directory. You can adjust the logging level in the `src/utils.py` file.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- [Sentence Transformers](https://www.sbert.net/)
- [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/)
- [LLaMA-2](https://github.com/facebookresearch/llama)
