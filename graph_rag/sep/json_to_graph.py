#!/usr/bin/env python3
"""
JSON to Neo4j Graph Converter
Converts structured PDF content to knowledge graph for RAG applications.

Two modes:
1. Hierarchical: Sections as nodes with parent-child relationships
2. Entity-rich: Extract entities and create fine-grained knowledge graph
"""

import json
import re
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from neo4j import GraphDatabase
import spacy
from collections import defaultdict, Counter

@dataclass
class Section:
    id: str
    title: str
    level: int
    parent_id: str
    text: str
    page_number: int
    is_toc_entry: bool

class Neo4jGraphBuilder:
    def __init__(self, uri: str, username: str, password: str):
        """Initialize Neo4j connection."""
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        
    def close(self):
        """Close Neo4j connection."""
        self.driver.close()
    
    def clear_database(self):
        """Clear all nodes and relationships."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("✓ Database cleared")

class HierarchicalGraphBuilder(Neo4jGraphBuilder):
    """Build simple hierarchical graph with sections as nodes."""
    
    def build_hierarchical_graph(self, sections: List[Section]):
        """Build hierarchical section graph."""
        print("Building hierarchical section graph...")
        
        with self.driver.session() as session:
            # Create section nodes
            for section in sections:
                session.run("""
                    CREATE (s:Section {
                        id: $id,
                        title: $title,
                        level: $level,
                        text: $text,
                        page_number: $page_number,
                        is_toc_entry: $is_toc_entry,
                        word_count: $word_count,
                        char_count: $char_count
                    })
                """, 
                id=section.id,
                title=section.title,
                level=section.level,
                text=section.text,
                page_number=section.page_number,
                is_toc_entry=section.is_toc_entry,
                word_count=len(section.text.split()),
                char_count=len(section.text)
                )
            
            # Create parent-child relationships
            for section in sections:
                if section.parent_id:
                    session.run("""
                        MATCH (parent:Section {id: $parent_id})
                        MATCH (child:Section {id: $child_id})
                        CREATE (parent)-[:HAS_SUBSECTION]->(child)
                        CREATE (child)-[:BELONGS_TO]->(parent)
                    """,
                    parent_id=section.parent_id,
                    child_id=section.id
                    )
            
            # Create sequential relationships (next/previous sections)
            sorted_sections = sorted(sections, key=lambda x: (x.level, x.id))
            for i in range(len(sorted_sections) - 1):
                current = sorted_sections[i]
                next_section = sorted_sections[i + 1]
                if current.level == next_section.level:  # Same level siblings
                    session.run("""
                        MATCH (current:Section {id: $current_id})
                        MATCH (next:Section {id: $next_id})
                        CREATE (current)-[:FOLLOWED_BY]->(next)
                    """,
                    current_id=current.id,
                    next_id=next_section.id
                    )
        
        print(f"✓ Created {len(sections)} section nodes with hierarchical relationships")

class EntityGraphBuilder(Neo4jGraphBuilder):
    """Build entity-rich knowledge graph with NLP extraction."""
    
    def __init__(self, uri: str, username: str, password: str):
        super().__init__(uri, username, password)
        # Load spaCy model (install with: python -m spacy download en_core_web_sm)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def extract_entities(self, text: str, title: str = "") -> Dict[str, List[str]]:
        """Extract entities from text using spaCy."""
        if not self.nlp:
            return self._simple_entity_extraction(text, title)
        
        # Combine title and text for entity extraction
        full_text = f"{title}. {text}" if title else text
        doc = self.nlp(full_text)
        
        entities = defaultdict(list)
        for ent in doc.ents:
            if len(ent.text.strip()) > 1:  # Filter out single characters
                entities[ent.label_].append(ent.text.strip())
        
        # Also extract key phrases (noun phrases)
        noun_phrases = [chunk.text.strip() for chunk in doc.noun_chunks 
                       if len(chunk.text.strip()) > 3 and len(chunk.text.split()) <= 4]
        
        if noun_phrases:
            entities['CONCEPT'] = noun_phrases[:10]  # Limit to top 10
        
        return dict(entities)
    
    def _simple_entity_extraction(self, text: str, title: str = "") -> Dict[str, List[str]]:
        """Fallback entity extraction without spaCy."""
        entities = defaultdict(list)
        
        # Extract potential concepts (capitalized phrases)
        full_text = f"{title}. {text}" if title else text
        
        # Find capitalized words/phrases
        concepts = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', full_text)
        concepts = [c for c in concepts if len(c) > 3 and len(c.split()) <= 3]
        
        # Find acronyms
        acronyms = re.findall(r'\b[A-Z]{2,}\b', full_text)
        
        if concepts:
            entities['CONCEPT'] = list(set(concepts))[:10]
        if acronyms:
            entities['ACRONYM'] = list(set(acronyms))[:5]
        
        return dict(entities)
    
    def build_entity_graph(self, sections: List[Section]):
        """Build entity-rich knowledge graph."""
        print("Building entity-rich knowledge graph...")
        
        all_entities = defaultdict(Counter)
        section_entities = {}
        
        # Extract entities from all sections
        for section in sections:
            entities = self.extract_entities(section.text, section.title)
            section_entities[section.id] = entities
            
            # Count entity occurrences across document
            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    all_entities[entity_type][entity] += 1
        
        with self.driver.session() as session:
            # Create section nodes
            for section in sections:
                session.run("""
                    CREATE (s:Section {
                        id: $id,
                        title: $title,
                        level: $level,
                        text: $text,
                        page_number: $page_number,
                        is_toc_entry: $is_toc_entry,
                        word_count: $word_count
                    })
                """, **section.__dict__, word_count=len(section.text.split()))
            
            # Create entity nodes (only frequent entities to avoid noise)
            for entity_type, entity_counter in all_entities.items():
                for entity, count in entity_counter.items():
                    if count >= 2:  # Only entities mentioned at least twice
                        session.run(f"""
                            CREATE (e:Entity:{entity_type} {{
                                name: $name,
                                type: $type,
                                frequency: $frequency,
                                normalized_name: $normalized_name
                            }})
                        """,
                        name=entity,
                        type=entity_type,
                        frequency=count,
                        normalized_name=entity.lower().strip()
                        )
            
            # Create section-entity relationships
            for section_id, entities in section_entities.items():
                for entity_type, entity_list in entities.items():
                    for entity in entity_list:
                        if all_entities[entity_type][entity] >= 2:  # Only frequent entities
                            session.run(f"""
                                MATCH (s:Section {{id: $section_id}})
                                MATCH (e:Entity:{entity_type} {{name: $entity_name}})
                                CREATE (s)-[:MENTIONS {{type: $entity_type}}]->(e)
                                CREATE (e)-[:MENTIONED_IN]->(s)
                            """,
                            section_id=section_id,
                            entity_name=entity,
                            entity_type=entity_type
                            )
            
            # Create hierarchical relationships for sections
            for section in sections:
                if section.parent_id:
                    session.run("""
                        MATCH (parent:Section {id: $parent_id})
                        MATCH (child:Section {id: $child_id})
                        CREATE (parent)-[:HAS_SUBSECTION]->(child)
                    """,
                    parent_id=section.parent_id,
                    child_id=section.id
                    )
            
            # Create entity co-occurrence relationships
            self._create_entity_cooccurrences(session, section_entities, all_entities)
        
        total_entities = sum(len(counter) for counter in all_entities.values())
        frequent_entities = sum(1 for counter in all_entities.values() 
                              for count in counter.values() if count >= 2)
        
        print(f"✓ Created {len(sections)} section nodes")
        print(f"✓ Extracted {total_entities} total entities")
        print(f"✓ Created {frequent_entities} frequent entity nodes")
    
    def _create_entity_cooccurrences(self, session, section_entities: Dict, all_entities: Dict):
        """Create relationships between entities that co-occur in sections."""
        print("Creating entity co-occurrence relationships...")
        
        for section_id, entities in section_entities.items():
            entity_pairs = []
            
            # Get all entities in this section
            all_section_entities = []
            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    if all_entities[entity_type][entity] >= 2:
                        all_section_entities.append((entity, entity_type))
            
            # Create pairs for co-occurrence
            for i, (entity1, type1) in enumerate(all_section_entities):
                for entity2, type2 in all_section_entities[i+1:]:
                    session.run(f"""
                        MATCH (e1:Entity:{type1} {{name: $entity1}})
                        MATCH (e2:Entity:{type2} {{name: $entity2}})
                        MERGE (e1)-[r:CO_OCCURS_WITH]-(e2)
                        ON CREATE SET r.frequency = 1
                        ON MATCH SET r.frequency = r.frequency + 1
                    """,
                    entity1=entity1,
                    entity2=entity2
                    )

def load_sections_from_json(file_path: str) -> List[Section]:
    """Load sections from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    sections = []
    for item in data:
        section = Section(
            id=item['id'],
            title=item['title'],
            level=item['level'],
            parent_id=item.get('parent_id', ''),
            text=item['text'],
            page_number=item['page_number'],
            is_toc_entry=item['is_toc_entry']
        )
        sections.append(section)
    
    return sections

def print_graph_stats(driver):
    """Print statistics about the created graph."""
    with driver.session() as session:
        # Section stats
        section_count = session.run("MATCH (s:Section) RETURN count(s) as count").single()['count']
        
        # Entity stats if they exist
        entity_result = session.run("MATCH (e:Entity) RETURN count(e) as count").single()
        entity_count = entity_result['count'] if entity_result else 0
        
        # Relationship stats
        rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()
        relationship_count = rel_result['count'] if rel_result else 0
        
        print(f"\n📊 Graph Statistics:")
        print(f"   Sections: {section_count}")
        if entity_count > 0:
            print(f"   Entities: {entity_count}")
        print(f"   Relationships: {relationship_count}")

def main():
    """Main function to build the graph."""
    
    # Configuration
    NEO4J_URI = "bolt://localhost:7687"  # Default Neo4j URI
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = "password"  # Change this to your Neo4j password
    JSON_FILE = "data/structured_content.json"
    
    print("🚀 JSON to Neo4j Graph Converter")
    print("=" * 50)
    
    # Load data
    print(f"Loading sections from {JSON_FILE}...")
    sections = load_sections_from_json(JSON_FILE)
    print(f"✓ Loaded {len(sections)} sections")
    
    # Choose graph type
    print("\nChoose graph type:")
    print("1. Hierarchical (Simple section-based graph)")
    print("2. Entity-rich (Advanced knowledge graph with NLP)")
    print("3. Both (Create both types)")
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    try:
        if choice in ['1', '3']:
            # Build hierarchical graph
            builder = HierarchicalGraphBuilder(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
            builder.clear_database()
            builder.build_hierarchical_graph(sections)
            print_graph_stats(builder.driver)
            builder.close()
        
        if choice in ['2', '3']:
            # Build entity-rich graph
            builder = EntityGraphBuilder(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
            if choice == '2':  # Only clear if not building both
                builder.clear_database()
            builder.build_entity_graph(sections)
            print_graph_stats(builder.driver)
            builder.close()
        
        print("\n✅ Graph creation completed!")
        print("\n🔍 Useful Cypher queries to explore your graph:")
        print("   # Find all top-level sections:")
        print("   MATCH (s:Section) WHERE s.level = 1 RETURN s.title, s.id")
        print("   ")
        print("   # Find sections with most text:")
        print("   MATCH (s:Section) RETURN s.title, s.word_count ORDER BY s.word_count DESC LIMIT 10")
        
        if choice in ['2', '3']:
            print("   ")
            print("   # Find most mentioned entities:")
            print("   MATCH (e:Entity) RETURN e.name, e.type, e.frequency ORDER BY e.frequency DESC LIMIT 10")
            print("   ")
            print("   # Find sections mentioning specific concept:")
            print("   MATCH (s:Section)-[:MENTIONS]->(e:Entity {name: 'Learning Suite'}) RETURN s.title")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure Neo4j is running and credentials are correct.")

if __name__ == "__main__":
    main()
