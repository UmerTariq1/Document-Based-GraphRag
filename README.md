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

## 🔧 Core Components

### 1. PDF Processing (`parse_pdf.py`)
- **Input**: PDF documents (e.g., technical documentation, manuals)
- **Output**: Structured JSON with hierarchical content
- **Features**: 
  - Extracts text, titles, and section hierarchies
  - Preserves document structure and page numbers
  - Handles table of contents and subsections

### 2. Graph Ingestion (`ingest.py`)
Transforms JSON data into a rich Neo4j knowledge graph with:

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

### 3. Query Engine (`query.py`)
Advanced GraphRAG query system featuring:

#### Multi-Modal Search:
- **Semantic Search**: Uses embeddings for content similarity
- **Keyword Matching**: Identifies sections with matching terms
- **Relationship Traversal**: Explores connected content

#### AI Integration:
- **OpenAI Integration**: Generates comprehensive responses using GPT models
- **Context Assembly**: Combines main content with related sections
- **Prompt Engineering**: Uses structured templates for optimal responses

#### Intelligent Filtering:
- Configurable similarity thresholds
- Minimum keyword match requirements
- Category-based result organization

### 4. Web Interface (`visualize.py`)
Interactive Streamlit application providing:

#### User Features:
- **Natural Language Querying**: Ask questions in plain English
- **Real-time Results**: Instant search with AI-generated responses
- **Relationship Visualization**: See how content pieces connect
- **Advanced Filtering**: Customize search parameters

#### Configuration Options:
- **Threshold Controls**: Adjust similarity and keyword matching requirements
- **Result Sorting**: Order by keyword matches or similarity scores
- **LLM Settings**: Choose models and response length
- **Keyword Exclusion**: Filter out unwanted terms

## 🚀 Getting Started

### Prerequisites
- Python 3.11
- Neo4j Database (running on `bolt://localhost:7687`)
- OpenAI API Key (for AI responses)

### Installation

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
   - Install and start Neo4j
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
   - Graph visualization query

#### Optional: Advanced Configuration
- Adjust thresholds in the sidebar
- Exclude specific keywords
- Choose different AI models
- Customize result sorting and filtering

## 🎮 Using the Interface


### Interface Features:

#### ⚙️ Query Settings:
- **Main Node Thresholds**: Control primary result requirements
- **Connected Node Thresholds**: Filter related content
- **Sorting Options**: Keyword matches vs. similarity scores
- **Result Limits**: Control number of displayed results

#### 🤖 AI Settings:
- **Model Selection**: Choose from GPT-4o, GPT-4o-mini, GPT-3.5-turbo
- **Response Length**: Customize AI response detail level
- **Enable/Disable**: Toggle AI responses on/off

#### 🔍 Advanced Features:
- **Keyword Exclusion**: Filter out common or unwanted terms
- **Relationship Details**: See exactly how content pieces connect
- **Graph Visualization**: Generate Cypher queries for Neo4j Browser

## 📊 Example Output

When you query "What is the imc Learning Suite?", you'll get:

1. **🤖 AI-Generated Response**: Comprehensive answer synthesized from multiple sources
2. **📄 Source Section**: Primary matching content with similarity scores
3. **📚 Connected Nodes**: Related sections with relationship explanations
4. **🔍 Graph Visualization**: Cypher query to explore in Neo4j Browser

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

## 📈 Use Cases

- **Technical Documentation**: Query complex software manuals
- **Research Papers**: Explore academic content and citations
- **Product Manuals**: Find specific features and procedures
- **Policy Documents**: Navigate regulatory and compliance materials
- **Educational Content**: Interactive learning from textbooks

## 🤝 Contributing

This project provides a foundation for document-based knowledge graphs. Contributions welcome for:
- Additional document formats (Word, HTML, etc.)
- Enhanced relationship types
- Improved UI/UX features
- Performance optimizations
- Additional LLM integrations

## 📝 License

[Add your license information here]

---

**Built with**: Python, Neo4j, Streamlit, OpenAI, SentenceTransformers, KeyBERT
