# Document-Based GraphRAG

A comprehensive **Graph-based Retrieval-Augmented Generation (GraphRAG)** system that transforms PDF documents into an intelligent knowledge graph and provides advanced querying capabilities through a user-friendly web interface.

## 🎯 Project Overview

This project implements a complete pipeline for document understanding and intelligent querying:

1. **PDF Processing** → **Knowledge Graph Creation** → **Intelligent Querying** → **Interactive Visualization**

The system extracts structured content from PDF documents, builds a semantic knowledge graph with multiple relationship types, and enables natural language querying with AI-powered responses.

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

### Usage Workflow

#### Step 1: Launch the Web Interface
```bash
streamlit run ui/visualize.py
```
- **Access**: Open `http://localhost:8501` in your browser
- **Features**: Two main tabs - Ingestion and Query

#### Step 2: Document Ingestion
1. Navigate to the **📁 Ingestion** tab
2. Upload your PDF document using the file uploader
3. Click "🚀 Start Ingestion" to process the document
4. The system will:
   - Extract document structure
   - Create knowledge graph in Neo4j
   - Generate semantic relationships
   - Show progress in real-time

#### Step 3: Query Your Documents
1. Switch to the **🔍 Query** tab
2. Enter your question in natural language
3. Use the sidebar to configure:
   - Similarity thresholds
   - Keyword matching requirements
   - AI response settings
   - Result filtering options
4. Click "🔍 Search" to get:
   - AI-generated response
   - Source content with relationships
   - Connected sections
   - Graph visualization query which you can enter in the neo4j application to visualize the graph



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
