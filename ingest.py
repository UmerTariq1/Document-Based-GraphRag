#!/usr/bin/env python3
"""
GraphRAG Ingestion Pipeline - Load structured PDF content into Neo4j with semantic relationships

Node Creation (create_section_nodes):
------------------------------------
Creates nodes in Neo4j based on section levels:
Level 1 → Section nodes
Level 2 → SubSection nodes
Level 3 → SubSubSection nodes
For each node, it:
Extracts keywords using KeyBERT
Generates embeddings using SentenceTransformer
Stores metadata (title, text, word count, etc.)



Relationship Creation:
----------------------

Hierarchical Relationships:
HAS_SUBSECTION: Parent to child relationship
PARENT: Child to parent relationship (inverse of HAS_SUBSECTION)
NEXT: Sequential relationship between sibling sections

Semantic Relationships:

KEYWORD_MENTIONS: Created when sections share keywords
Property: keyword_match_count (number of matching keywords)

SEMANTIC_SIMILAR_TO: Created based on embedding similarity
Property: similarity_score (cosine similarity between 0-1)
Only created when similarity exceeds threshold (default 0.85)

"""

import json
from neo4j import GraphDatabase
from typing import List, Dict
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm

class GraphRAGIngestion:
    def __init__(self, uri: str = "bolt://localhost:7687", username: str = "neo4j", password: str = "password"):
        """Initialize Neo4j connection and ML models."""
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        print("✓ Connected to Neo4j")
        
        # Initialize ML models
        print("🤖 Loading ML models...")
        self.keyword_model = KeyBERT()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ ML models loaded")
    
    def close(self):
        """Close Neo4j connection."""
        self.driver.close()
        print("✓ Connection closed")
    
    def clear_database(self):
        """Clear all existing data."""
        print("🧹 Clearing database...")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("✓ Database cleared")
    
    def create_indexes(self):
        """Create indexes for better query performance."""
        print("🔍 Creating indexes...")
        with self.driver.session() as session:
            # Create indexes for each section type
            for label in ["Section", "SubSection", "SubSubSection"]:
                session.run(f"CREATE INDEX {label.lower()}_id_index IF NOT EXISTS FOR (s:{label}) ON (s.id)")
                session.run(f"CREATE INDEX {label.lower()}_level_index IF NOT EXISTS FOR (s:{label}) ON (s.level)")
                session.run(f"CREATE TEXT INDEX {label.lower()}_text_index IF NOT EXISTS FOR (s:{label}) ON (s.text)")
                session.run(f"CREATE TEXT INDEX {label.lower()}_title_index IF NOT EXISTS FOR (s:{label}) ON (s.title)")
        print("✓ Indexes created")
    
    def get_graph_stats(self):
        """Display statistics about the graph."""
        print("\n📊 Graph Statistics:")
        print("=" * 50)
        
        with self.driver.session() as session:
            # Get node counts by type
            result = session.run("""
                MATCH (n)
                RETURN labels(n) as type, count(*) as count
                ORDER BY count DESC
            """)
            
            print("\nNode Counts:")
            for record in result:
                node_type = record['type'][0] if record['type'] else 'Unknown'
                count = record['count']
                print(f"  • {node_type}: {count}")
            
            # Get relationship counts by type
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as type, count(*) as count
                ORDER BY count DESC
            """)
            
            print("\nRelationship Counts:")
            for record in result:
                rel_type = record['type']
                count = record['count']
                print(f"  • {rel_type}: {count}")
            
            # Get average node properties
            result = session.run("""
                MATCH (n)
                RETURN 
                    avg(size(n.text)) as avg_text_length,
                    avg(n.word_count) as avg_word_count,
                    avg(size(n.keywords)) as avg_keywords
            """)
            
            stats = result.single()
            print("\nAverage Properties:")
            print(f"  • Text Length: {int(stats['avg_text_length'])} characters")
            print(f"  • Word Count: {int(stats['avg_word_count'])} words")
            print(f"  • Keywords: {int(stats['avg_keywords'])} per node")
    
    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """Extract keywords from text using KeyBERT."""
        keywords = self.keyword_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words='english',
            top_n=top_n
        )
        # print("Keywords : ", keywords)
        return [keyword for keyword, score in keywords]
    
    def create_section_nodes(self, sections: List[Dict]):
        """Create Section nodes in Neo4j with keywords and embeddings."""
        print("🏗️  Creating section nodes...")
        with self.driver.session() as session:
            for section in tqdm(sections, desc="Creating section nodes"):
                # Calculate word count
                word_count = len(section.get('text', '').split()) if section.get('text') else 0
                
                # Determine label based on level
                level = section.get('level', 1)
                if level == 1:
                    label = "Section"
                elif level == 2:
                    label = "SubSection"
                else:
                    label = "SubSubSection"
                
                # Extract keywords
                keywords = self.extract_keywords(section.get('text', ''))
                
                # Generate embedding
                embedding = self.embedding_model.encode(section.get('text', '')).tolist()
                
                session.run(f"""
                    CREATE (s:{label} {{
                        id: $id,
                        title: $title,
                        text: $text,
                        level: $level,
                        page_number: $page_number,
                        is_toc_entry: $is_toc_entry,
                        word_count: $word_count,
                        char_count: $char_count,
                        keywords: $keywords,
                        embedding: $embedding
                    }})
                """, 
                id=section['id'],
                title=section.get('title', ''),
                text=section.get('text', ''),
                level=level,
                page_number=section.get('page_number', 1),
                is_toc_entry=section.get('is_toc_entry', False),
                word_count=word_count,
                char_count=len(section.get('text', '')),
                keywords=keywords,
                embedding=embedding
                )
        
        print(f"✓ Created {len(sections)} section nodes")
    
    def create_hierarchical_relationships(self, sections: List[Dict]):
        """Create bidirectional HAS_SUBSECTION relationships between sections."""
        print("🔗 Creating hierarchical relationships...")
        
        relationship_count = 0
        with self.driver.session() as session:
            for section in sections:
                parent_id = section.get('parent_id')
                if parent_id:
                    result = session.run("""
                        MATCH (parent) WHERE parent.id = $parent_id
                        MATCH (child) WHERE child.id = $child_id
                        CREATE (parent)-[:HAS_SUBSECTION]->(child)
                        CREATE (child)-[:PARENT]->(parent)
                        RETURN parent.id, child.id
                    """,
                    parent_id=parent_id,
                    child_id=section['id']
                    )
                    if result.single():
                        relationship_count += 1
        
        print(f"✓ Created {relationship_count} hierarchical relationships")
    
    def create_sibling_relationships(self, sections: List[Dict]):
        """Create NEXT relationships between sibling sections."""
        print("➡️  Creating sibling relationships...")
        
        # Group sections by parent_id
        siblings = {}
        for section in sections:
            parent_id = section.get('parent_id')
            if parent_id:
                if parent_id not in siblings:
                    siblings[parent_id] = []
                siblings[parent_id].append(section)
        
        # Sort siblings by id and create NEXT relationships
        relationship_count = 0
        with self.driver.session() as session:
            for parent_id, sibling_group in siblings.items():
                # Sort siblings by id
                sibling_group.sort(key=lambda x: x['id'])
                
                # Create NEXT relationships
                for i in range(len(sibling_group) - 1):
                    result = session.run("""
                        MATCH (current) WHERE current.id = $current_id
                        MATCH (next) WHERE next.id = $next_id
                        CREATE (current)-[:NEXT]->(next)
                        RETURN current.id, next.id
                    """,
                    current_id=sibling_group[i]['id'],
                    next_id=sibling_group[i + 1]['id']
                    )
                    if result.single():
                        relationship_count += 1
        
        print(f"✓ Created {relationship_count} sibling relationships")
    
    def create_mention_relationships(self, sections: List[Dict], min_keyword_matches: int = 1):
        """Create bidirectional KEYWORD_MENTIONS relationships based on keyword matches."""
        print("🔍 Creating keyword mention relationships...")
        
        relationship_count = 0
        with self.driver.session() as session:
            for section in sections:
                # Get keywords for this section
                result = session.run("""
                    MATCH (s) WHERE s.id = $section_id
                    RETURN s.keywords as keywords
                """, section_id=section['id'])
                
                keywords = result.single()['keywords']
                if not keywords:  # Skip if no keywords
                    continue

                # Find sections with matching keywords
                matches = session.run("""
                    MATCH (target)
                    WHERE target.id <> $section_id
                    AND target.keywords IS NOT NULL
                    AND size(target.keywords) > 0
                    RETURN target.id, target.keywords as target_keywords
                """, section_id=section['id'])
                
                for match in matches:
                    # Count matching keywords between source and target
                    target_keywords = match['target_keywords']
                    matching_keywords = set(kw.lower() for kw in keywords) & set(kw.lower() for kw in target_keywords)
                    keyword_match_count = len(matching_keywords)
                    
                    # Only create relationship if there are enough matching keywords
                    if keyword_match_count >= min_keyword_matches:
                        session.run("""
                            MATCH (source) WHERE source.id = $source_id
                            MATCH (target) WHERE target.id = $target_id
                            MERGE (source)-[r:KEYWORD_MENTIONS {bidirectional: true}]->(target)
                            SET r.keyword_match_count = $keyword_match_count
                        """, source_id=section['id'], target_id=match['target.id'], keyword_match_count=keyword_match_count)
                        relationship_count += 1

                        
        
        print(f"✓ Created {relationship_count} keyword mention relationships")
    
    def create_similarity_relationships(self, sections: List[Dict], similarity_threshold: float = 0.60):
        """Create bidirectional SEMANTIC_SIMILAR_TO relationships based on embedding similarity."""
        print("🔄 Creating semantic similarity relationships...")
        
        # Get all embeddings
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s)
                RETURN s.id as id, s.embedding as embedding
            """)
            
            # Create embedding matrix
            embeddings = {}
            for record in result:
                embeddings[record['id']] = np.array(record['embedding'])
            
            # Calculate similarities and create relationships
            relationship_count = 0
            for id1, emb1 in embeddings.items():
                for id2, emb2 in embeddings.items():
                    if id1 != id2:  # Avoid self-loops
                        similarity = cosine_similarity([emb1], [emb2])[0][0]
                        if similarity >= similarity_threshold:
                            session.run("""
                                MATCH (s1) WHERE s1.id = $id1
                                MATCH (s2) WHERE s2.id = $id2
                                MERGE (s1)-[r:SEMANTIC_SIMILAR_TO {bidirectional: true}]->(s2)
                                SET r.similarity_score = $similarity
                            """, id1=id1, id2=id2, similarity=float(similarity))
                            relationship_count += 1
        
        print(f"✓ Created {relationship_count} semantic similarity relationships")
    
    def load_json_data(self, json_file_path: str) -> List[Dict]:
        """Load and parse JSON data from file.
        
        Args:
            json_file_path: Path to the JSON file containing section data
            
        Returns:
            List of dictionaries containing section data
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Extract the sections array from the top-level object
            sections = data.get('sections', [])
            print(f"✓ Loaded {len(sections)} sections from {json_file_path}")
            return sections
        except FileNotFoundError:
            print(f"❌ File not found: {json_file_path}")
            return []
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON format in file: {json_file_path}")
            return []
        except Exception as e:
            print(f"❌ Error loading JSON data: {e}")
            return []
    
    def ingest_data(self, json_file_path: str):
        """Main ingestion process."""
        print("🚀 Starting GraphRAG Ingestion Process")
        print("=" * 50)
        
        # Load data
        sections = self.load_json_data(json_file_path)
        if not sections:
            print("❌ No data to ingest")
            return
        
        # Clear existing data
        self.clear_database()
        
        # Create graph structure
        self.create_section_nodes(sections)
        self.create_hierarchical_relationships(sections)
        self.create_sibling_relationships(sections)
        self.create_mention_relationships(sections)
        self.create_similarity_relationships(sections)
        self.create_indexes()
        
        # Show results
        self.get_graph_stats()
        
        print("\n✅ Ingestion completed successfully!")

def main():
    """Main function with hard-coded file path."""
    JSON_FILE_PATH = "data/structured_content.json"
    
    # Initialize ingestion
    ingestion = GraphRAGIngestion()
    
    try:
        # Run ingestion process
        ingestion.ingest_data(JSON_FILE_PATH)
        
    except Exception as e:
        print(f"❌ Error during ingestion: {e}")
    
    finally:
        ingestion.close()

if __name__ == "__main__":
    main()