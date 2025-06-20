# Document-Based GraphRAG

A comprehensive **Graph-based Retrieval-Augmented Generation (GraphRAG)** system that transforms PDF documents into an intelligent knowledge graph and provides advanced querying capabilities through a user-friendly web interface.

## 🎯 Project Overview

This project implements a complete pipeline for document understanding and intelligent querying:

1. **PDF Processing** → **Knowledge Graph Creation** → **Intelligent Querying** → **Interactive Visualization**

The system extracts structured content from PDF documents, builds a semantic knowledge graph with multiple relationship types, and enables natural language querying with AI-powered responses.

## 📦 Project structure

```text
.
├── data/                 # Example PDFs, images, JSON output
├── logs/                 # Runtime logs
├── src/                  # All application code
│   ├── ingest.py         # CLI ingestion entry-point
│   ├── query.py          # Query/LLM orchestration entry-point
│   ├── parse_pdf.py      # Stand-alone PDF-to-JSON script
│   ├── ui/               # Streamlit UI
│   ├── utils/            # Shared helpers (ML models, OpenAI, Neo4j, text)
│   ├── pdf_reader/       # PDF parsing library
│   └── graph_rag/        # Experimental graph utilities
└── requirements.txt      # Python dependencies
```


## 🖼️ Interface Preview

<p align="center">
  <img src="data/ref_imgs/ingestion%20page.png" alt="Ingestion Page" width="800"/>
  <br/>
  <em>Ingestion tab – upload & processing workflow</em>
  <br/><br/>
  <img src="data/ref_imgs/query%20page%20-%20without%20answer.png" alt="Query Page – No Data" width="800"/>
  <br/>
  <em>Query tab – When user has not asked the query yet</em>
  <br/><br/>
  <img src="data/ref_imgs/query%20page%20-%20with%20answer.png" alt="Query Page – Answer" width="800"/>
  <br/>
  <em>Query tab – example result with AI-generated answer and graph context</em>
</p>

## 🏗️ System Architecture

```
📄 PDF Document
    ↓ (parse_pdf.py)
📋 JSON Structure  
    ↓ (ingest.py)
🕸️ Neo4j Knowledge Graph
    ↓ (query.py + visualize.py)
🖥️ Interactive Web Interface
```

## 🔧 Graph Components


#### Node Types:
- **Section** (Level 1): Main document sections
- **SubSection** (Level 2): Subsections within main sections  
- **SubSubSection** (Level 3): Detailed subsections

#### Node Properties:
- **Content**: Title, text, page number, word count
- **Keywords**: Extracted using KeyBERT for semantic understanding
- **Embeddings**: Generated using SentenceTransformer for similarity calculations

#### Relationship Types:
- **`HAS_SUBSECTION`**: Hierarchical parent → child relationships
- **`PARENT`**: Inverse child → parent relationships
- **`NEXT`**: Sequential relationships between sibling sections
- **`KEYWORD_MENTIONS`**: Connections based on shared keywords
- **`SEMANTIC_SIMILAR_TO`**: Relationships based on embedding similarity

## 🔄 Application Flow

### Query Process Flow

The query interface implements a sophisticated two-stage retrieval process for optimal results:

1. **Initial Candidate Retrieval**: When you submit a query, the system first retrieves multiple candidate sections (configurable from 1-20) that match your query based on either keyword matches or semantic similarity scores.

2. **Interactive Candidate Selection**: If multiple candidates are found, you'll see an expandable list showing each candidate section with:
   - Section title and ID
   - Page number
   - Keyword match count and similarity score
   - Preview of the section content (first 600 characters)

3. **Main Answer Generation**: After selecting your preferred candidate, the system:
   - Uses the chosen section as the primary answer
   - Finds related content through graph relationships (parent sections, subsections, keyword mentions, semantic similarities)
   - Here you still have the option to explore other candidiate sections.
   
This approach ensures you have control over which section serves as the main answer while still benefiting from the rich contextual information provided by the knowledge graph connections.

### 3. Web Interface (Streamlit application)
Interactive Streamlit application providing:

#### User Features:
- **Natural Language Querying**: Ask questions in plain English
- **Real-time Results**: Instant search with AI-generated responses
- **Relationship Visualization**: See how content pieces connect
- **Advanced Search parameters**: Customize search parameters

#### Configuration Options:
- **Threshold Controls**: Adjust similarity and keyword matching requirements
- **Result Sorting**: Order by keyword matches or similarity scores
- **LLM Settings**: Choose models and response length
- **Keyword Exclusion**: Filter out unwanted terms which wont be used for keyword matching

## 🚀 Getting Started

### Prerequisites
- Python 3.11
- libraries as mentioned in requirements.txt
- Neo4j Database (running on `bolt://localhost:7687`)
- OpenAI API Key (for AI responses)


### Environment Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd Document-Based-GraphRag
```

2. **Create Virtual Environment and activate it**:
```bash
python3.11 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**:
```bash
# Create .env file
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

4. **Start Neo4j database**:
   - Install and start Neo4j project and database
   - Default credentials: `neo4j/password`
   - Ensure it's running on `bolt://localhost:7687`

4. **Launch the Web Interface**:
```bash
streamlit run src/ui/visualize.py
```
- **Access**: Open `http://localhost:8501` in your browser
- **Features**: Two main tabs - Ingestion and Query



## 🛠️ Technical Details

### Machine Learning Models:
- **KeyBERT**: Keyword extraction for semantic understanding
- **SentenceTransformer** (`all-MiniLM-L6-v2`): Embedding generation
- **OpenAI GPT**: Response generation and synthesis

## 🔧 Configuration

### Default Thresholds:
- **Main Node Similarity**: 0.7
- **Connected Node Similarity**: 0.7  
- **Minimum Keywords**: 2 matches
- **Semantic Similarity**: 0.95 for relationships

### Customization:
All thresholds and settings can be adjusted through the web interface or by modifying the configuration parameters in the code.

## ➡️ Future TODOs
- Support multiple documents ingestion and their connection 