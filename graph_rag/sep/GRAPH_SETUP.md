# Graph RAG Setup Guide

Transform your structured PDF content into a powerful knowledge graph for RAG applications.

## 🎯 Two Approaches Available

### 1. **Hierarchical Graph** (Simple)
- **Nodes**: Document sections as nodes
- **Relationships**: Parent-child hierarchy, sequential flow
- **Best for**: Document navigation, structure-aware search
- **Query examples**: "Find all subsections of chapter 2", "What comes after section 1.3?"

### 2. **Entity-Rich Graph** (Advanced) 
- **Nodes**: Sections + extracted entities (people, concepts, organizations)
- **Relationships**: Hierarchical + entity mentions + co-occurrences
- **Best for**: Concept-based search, entity relationships, knowledge discovery
- **Query examples**: "Find all sections mentioning 'Learning Suite'", "What concepts are related to 'imc'?"

## 🚀 Quick Start

### Prerequisites
```bash
# Install Neo4j (macOS with Homebrew)
brew install neo4j

# Or download from: https://neo4j.com/download/

# Install Python dependencies
pip install -r requirements_graph.txt

# Download spaCy model for NLP (needed for entity extraction)
python -m spacy download en_core_web_sm
```

### Start Neo4j
```bash
# Start Neo4j server
neo4j start

# Access Neo4j Browser at: http://localhost:7474
# Default credentials: username=neo4j, password=neo4j (change on first login)
```

### Build Your Graph
```bash
# Run the graph builder
python json_to_graph.py

# Choose your approach:
# 1 = Simple hierarchical graph
# 2 = Advanced entity-rich graph  
# 3 = Both (recommended)
```

### Query Your Graph
```bash
# Interactive RAG interface
python graph_rag_query.py

# Or use Neo4j Browser directly at http://localhost:7474
```

## 📊 Graph Structure

### Hierarchical Graph
```
(Section:1) -[:HAS_SUBSECTION]-> (Section:1.1)
(Section:1.1) -[:BELONGS_TO]-> (Section:1)
(Section:1.1) -[:FOLLOWED_BY]-> (Section:1.2)
```

### Entity-Rich Graph
```
(Section:1.1) -[:MENTIONS]-> (Entity:CONCEPT {name: "Learning Suite"})
(Entity:CONCEPT) -[:CO_OCCURS_WITH]-> (Entity:ORG {name: "imc"})
(Entity:CONCEPT) -[:MENTIONED_IN]-> (Section:1.1)
```

## 🔍 Example Queries

### Basic Navigation
```cypher
// Find all top-level sections
MATCH (s:Section) WHERE s.level = 1 
RETURN s.title, s.page_number ORDER BY s.id

// Get section hierarchy
MATCH (parent:Section)-[:HAS_SUBSECTION*]->(child:Section {id: "1.2"})
RETURN parent.title, child.title
```

### Content Search
```cypher
// Find sections containing keywords
MATCH (s:Section)
WHERE s.text CONTAINS "learning" OR s.title CONTAINS "learning"
RETURN s.id, s.title, s.page_number

// Sections with most content
MATCH (s:Section)
RETURN s.id, s.title, s.word_count
ORDER BY s.word_count DESC LIMIT 10
```

### Entity-Based Queries (Advanced Graph Only)
```cypher
// Most frequently mentioned entities
MATCH (e:Entity)
RETURN e.name, e.type, e.frequency
ORDER BY e.frequency DESC LIMIT 10

// Find sections mentioning specific concept
MATCH (s:Section)-[:MENTIONS]->(e:Entity {name: "Learning Suite"})
RETURN s.id, s.title, s.page_number

// Entity co-occurrences
MATCH (e1:Entity {name: "imc"})-[:CO_OCCURS_WITH]-(e2:Entity)
RETURN e2.name, e2.type
ORDER BY e2.frequency DESC
```

### RAG-Style Queries
```cypher
// Context-aware search
MATCH (s:Section)
WHERE s.text =~ "(?i).*learning.*system.*"
OPTIONAL MATCH (s)-[:MENTIONS]->(e:Entity)
WITH s, collect(e.name) as entities
RETURN s.id, s.title, s.text, entities
ORDER BY size(entities) DESC

// Find related sections through shared entities
MATCH (s1:Section)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(s2:Section)
WHERE s1.id = "1.2" AND s1 <> s2
RETURN s2.id, s2.title, collect(e.name) as shared_entities
```

## 🛠️ Configuration

### Connection Settings
Edit the connection parameters in both scripts:
```python
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"  
NEO4J_PASSWORD = "your_password"  # Change this!
JSON_FILE = "data/structured_content.json"
```

### Entity Extraction Tuning
In `json_to_graph.py`, adjust these parameters:
```python
# Minimum frequency for entities to be included
if count >= 2:  # Only entities mentioned at least twice

# Limit noun phrases extracted
entities['CONCEPT'] = noun_phrases[:10]  # Top 10 concepts per section
```

## 🎯 Use Cases

### 1. Document Q&A
```python
# Query: "What is the imc Learning Suite?"
results = querier.semantic_search_sections("imc Learning Suite")
# Returns: Relevant sections with context
```

### 2. Knowledge Discovery  
```python
# Query: "What concepts are related to learning?"
results = querier.find_related_entities("learning")
# Returns: Co-occurring entities and their relationships
```

### 3. Navigation Assistant
```python
# Query: "Show me the structure of chapter 1"
hierarchy = querier.get_section_hierarchy("1")
# Returns: Parent, children, and sibling sections
```

## 🔧 Troubleshooting

### Common Issues

**Neo4j Connection Error**
- Ensure Neo4j is running: `neo4j status`
- Check credentials in scripts
- Verify port 7687 is not blocked

**spaCy Model Missing**
- Install: `python -m spacy download en_core_web_sm`
- Or disable NLP: Script will fall back to simple regex extraction

**Memory Issues with Large Documents**
- Increase Neo4j memory in `neo4j.conf`:
  ```
  dbms.memory.heap.initial_size=1G
  dbms.memory.heap.max_size=2G
  ```

**Poor Entity Extraction**
- Tune frequency threshold (currently 2)
- Adjust noun phrase limits
- Consider using larger spaCy models

## 📈 Performance Tips

1. **Index Creation**: After building graph, create indexes:
   ```cypher
   CREATE INDEX FOR (s:Section) ON (s.id)
   CREATE INDEX FOR (e:Entity) ON (e.name)
   CREATE TEXT INDEX FOR (s:Section) ON (s.text)
   ```

2. **Batch Processing**: For large documents, process in batches
3. **Memory Tuning**: Adjust Neo4j heap size based on document size
4. **Query Optimization**: Use EXPLAIN/PROFILE for slow queries

## 🔮 Next Steps

- **Vector Embeddings**: Add sentence embeddings for semantic similarity
- **Advanced NLP**: Use named entity linking, relation extraction
- **Multi-Document**: Extend to handle multiple documents with cross-references
- **LLM Integration**: Connect with GPT/Claude for answer generation
- **Web Interface**: Build a web UI for easier querying

## 📚 Resources

- [Neo4j Documentation](https://neo4j.com/docs/)
- [Cypher Query Language](https://neo4j.com/docs/cypher-manual/current/)
- [spaCy NLP Library](https://spacy.io/usage)
- [Graph RAG Patterns](https://neo4j.com/use-cases/real-time-recommendation-engine/) 