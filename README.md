# ClipQuery - Video RAG System

A Retrieval-Augmented Generation (RAG) system for searching through video transcripts using semantic similarity.

## Features

- **Dynamic Embedding Generation**: Automatically generates embeddings from transcript files instead of using pre-computed embeddings
- **Semantic Search**: Uses sentence transformers to find relevant video segments based on natural language queries
- **Timestamp Tracking**: Provides exact timestamps for each relevant segment
- **Interactive Query Interface**: Allows users to input custom queries
- **Batch Query Processing**: Run multiple example queries at once

## How It Works

1. **Transcript Processing**: The system reads `.txt` files from the `Max Life Videos` folder
2. **Segment Extraction**: Parses timestamped segments in the format `[start_time - end_time] text`
3. **Embedding Generation**: Uses the `all-MiniLM-L6-v2` model to generate embeddings for each text segment
4. **Vector Storage**: Stores embeddings in a ChromaDB vector database for fast similarity search
5. **Query Processing**: Converts user queries to embeddings and finds the most similar segments

## File Structure

```
ClipQuery/
├── rag_pipeline.py          # Main RAG pipeline implementation
├── example_rag.py           # Example queries script
├── interactive_query.py     # Interactive query interface
├── Max Life Videos/         # Folder containing transcript files
│   ├── Video 1.txt
│   ├── Video 2.txt
│   ├── Video 4.txt
│   ├── Video 5.txt
│   └── Test1.txt
└── vector_store/            # ChromaDB vector database (auto-generated)
```

## Usage

### 1. Run Example Queries

```bash
python example_rag.py
```

This will run 10 pre-defined queries relevant to the Max Life Videos content, including:
- How to qualify leads in bank assurance?
- What is the NOPP criteria?
- How to build rapport with customers?
- What are the benefits of insurance?
- And more...

### 2. Interactive Query Interface

```bash
python interactive_query.py
```

This provides an interactive interface where you can:
- Type your own custom queries
- Type `help` to see example queries
- Type `quit` or `exit` to stop

### 3. Programmatic Usage

```python
from rag_pipeline import VideoRAG

# Initialize the RAG system
rag = VideoRAG()

# Query for relevant segments
results = rag.query_videos("How to qualify leads in bank assurance?", n_results=5)

# Process results
for result in results:
    print(f"Video: {result['video_id']}")
    print(f"Timestamp: {result['timestamp']}")
    print(f"Text: {result['text']}")
    print(f"Similarity: {1 - result['similarity']:.4f}")
```

## Transcript File Format

The system expects transcript files in the following format:

```
[0.00 - 10.00]  Have you ever wondered how to effectively qualify leads in bank assurance?
[10.00 - 13.00]  Let's explore some real-life scenarios together.
[16.00 - 19.00]  First, consider the importance of building rapport.
```

Each line should contain:
- Timestamp in format `[start_time - end_time]`
- Text content after the timestamp

## Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `chromadb` - Vector database
- `sentence-transformers` - Embedding generation
- `torch` - PyTorch for ML operations

## Configuration

- **Embedding Model**: Uses `all-MiniLM-L6-v2` for generating embeddings
- **Vector Database**: ChromaDB with cosine similarity
- **Transcript Folder**: `Max Life Videos` (configurable in `rag_pipeline.py`)
- **Results**: Default 5 results per query (configurable)

## Performance

- **Initialization**: Takes a few seconds to process all transcript files and generate embeddings
- **Query Speed**: Sub-second response times for similarity search
- **Memory Usage**: Efficient storage using ChromaDB's optimized vector storage

## Customization

To modify the system:

1. **Change Embedding Model**: Update the model name in `VectorStore.__init__()`
2. **Adjust Results Count**: Modify the `n_results` parameter in query methods
3. **Add New Transcripts**: Simply add new `.txt` files to the `Max Life Videos` folder
4. **Custom Queries**: Modify the queries list in `example_rag.py` or use the interactive interface

## Troubleshooting

- **No Transcripts Found**: Ensure `.txt` files exist in the `Max Life Videos` folder
- **Encoding Issues**: Transcript files should be UTF-8 encoded
- **Memory Issues**: For large transcript collections, consider processing files in batches
