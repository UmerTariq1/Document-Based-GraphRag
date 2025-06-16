#!/usr/bin/env python3
"""
Graph RAG Query Interface
Query the Neo4j knowledge graph for document-based question answering.
"""

from neo4j import GraphDatabase
from typing import List, Dict, Any
import json

class GraphRAGQuerier:
    def __init__(self, uri: str, username: str, password: str):
        """Initialize Neo4j connection."""
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
    
    def close(self):
        """Close Neo4j connection."""
        self.driver.close()
    
    def find_relevant_sections_by_keywords(self, keywords: List[str], limit: int = 5) -> List[Dict[str, Any]]:
        """Find sections containing specific keywords."""
        keyword_pattern = '|'.join([f'(?i).*{kw}.*' for kw in keywords])
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:Section)
                WHERE s.title =~ $pattern OR s.text =~ $pattern
                RETURN s.id, s.title, s.text, s.page_number, s.level, s.word_count
                ORDER BY s.word_count DESC
                LIMIT $limit
            """, pattern=keyword_pattern, limit=limit)
            
            return [dict(record) for record in result]
    
    def find_sections_by_entity(self, entity_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find sections that mention a specific entity."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:Section)-[:MENTIONS]->(e:Entity)
                WHERE e.name CONTAINS $entity_name OR e.normalized_name CONTAINS $entity_name
                RETURN s.id, s.title, s.text, s.page_number, e.name as entity, e.type as entity_type
                ORDER BY s.word_count DESC
                LIMIT $limit
            """, entity_name=entity_name.lower(), limit=limit)
            
            return [dict(record) for record in result]
    
    def get_section_hierarchy(self, section_id: str) -> Dict[str, Any]:
        """Get the hierarchical context of a section (parent and children)."""
        with self.driver.session() as session:
            # Get the section itself
            section_result = session.run("""
                MATCH (s:Section {id: $section_id})
                RETURN s.id, s.title, s.text, s.page_number, s.level
            """, section_id=section_id)
            
            section = dict(section_result.single()) if section_result.single() else None
            if not section:
                return {}
            
            # Get parent
            parent_result = session.run("""
                MATCH (parent:Section)-[:HAS_SUBSECTION]->(s:Section {id: $section_id})
                RETURN parent.id, parent.title
            """, section_id=section_id)
            
            parent = dict(parent_result.single()) if parent_result.single() else None
            
            # Get children
            children_result = session.run("""
                MATCH (s:Section {id: $section_id})-[:HAS_SUBSECTION]->(child:Section)
                RETURN child.id, child.title
                ORDER BY child.id
            """, section_id=section_id)
            
            children = [dict(record) for record in children_result]
            
            return {
                'section': section,
                'parent': parent,
                'children': children
            }
    
    def find_related_entities(self, entity_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find entities that co-occur with the given entity."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e1:Entity)-[r:CO_OCCURS_WITH]-(e2:Entity)
                WHERE e1.name CONTAINS $entity_name
                RETURN e2.name, e2.type, r.frequency
                ORDER BY r.frequency DESC
                LIMIT $limit
            """, entity_name=entity_name, limit=limit)
            
            return [dict(record) for record in result]
    
    def semantic_search_sections(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Perform semantic search on sections (simple keyword-based for now)."""
        # Split query into keywords
        keywords = [word.strip().lower() for word in query.split() if len(word.strip()) > 2]
        
        # Remove common stop words
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}
        keywords = [kw for kw in keywords if kw not in stop_words]
        
        if not keywords:
            return []
        
        # Create search pattern
        keyword_pattern = '.*(' + '|'.join(keywords) + ').*'
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:Section)
                WHERE s.title =~ $pattern OR s.text =~ $pattern
                OPTIONAL MATCH (s)-[:MENTIONS]->(e:Entity)
                WITH s, count(e) as entity_count
                RETURN s.id, s.title, s.text, s.page_number, s.level, s.word_count, entity_count
                ORDER BY entity_count DESC, s.word_count DESC
                LIMIT $limit
            """, pattern=f'(?i){keyword_pattern}', limit=limit)
            
            return [dict(record) for record in result]
    
    def get_document_overview(self) -> Dict[str, Any]:
        """Get an overview of the document structure."""
        with self.driver.session() as session:
            # Top-level sections
            top_sections = session.run("""
                MATCH (s:Section) WHERE s.level = 1
                RETURN s.id, s.title, s.page_number
                ORDER BY s.id
            """)
            
            # Most mentioned entities
            top_entities = session.run("""
                MATCH (e:Entity)
                RETURN e.name, e.type, e.frequency
                ORDER BY e.frequency DESC
                LIMIT 10
            """)
            
            # Document stats
            stats = session.run("""
                MATCH (s:Section)
                WITH count(s) as total_sections,
                     sum(s.word_count) as total_words,
                     avg(s.word_count) as avg_words
                RETURN total_sections, total_words, avg_words
            """).single()
            
            return {
                'top_sections': [dict(record) for record in top_sections],
                'top_entities': [dict(record) for record in top_entities],
                'stats': dict(stats) if stats else {}
            }

def format_search_results(results: List[Dict[str, Any]], query: str = "") -> str:
    """Format search results for display."""
    if not results:
        return "No relevant sections found."
    
    output = []
    if query:
        output.append(f"🔍 Search Results for: '{query}'")
        output.append("=" * 50)
    
    for i, result in enumerate(results, 1):
        output.append(f"\n{i}. Section {result.get('s.id', result.get('id', 'Unknown'))}: {result.get('s.title', result.get('title', 'Untitled'))}")
        output.append(f"   📄 Page: {result.get('s.page_number', result.get('page_number', 'Unknown'))}")
        
        text = result.get('s.text', result.get('text', ''))
        if text:
            # Show first 200 characters
            preview = text[:200] + "..." if len(text) > 200 else text
            output.append(f"   📝 Preview: {preview}")
        
        if 'entity' in result:
            output.append(f"   🏷️  Mentions: {result['entity']} ({result.get('entity_type', 'Unknown')})")
    
    return "\n".join(output)

def main():
    """Interactive RAG query interface."""
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = "password"  # Change to your password
    
    print("🤖 Graph RAG Query Interface")
    print("=" * 50)
    
    try:
        querier = GraphRAGQuerier(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
        
        # Show document overview
        print("📊 Document Overview:")
        overview = querier.get_document_overview()
        
        if overview.get('stats'):
            stats = overview['stats']
            print(f"   Total Sections: {stats.get('total_sections', 0)}")
            print(f"   Total Words: {stats.get('total_words', 0):,}")
            print(f"   Average Words per Section: {stats.get('avg_words', 0):.1f}")
        
        if overview.get('top_sections'):
            print(f"\n📋 Top-level Sections:")
            for section in overview['top_sections'][:5]:
                print(f"   {section['s.id']}: {section['s.title']} (page {section['s.page_number']})")
        
        if overview.get('top_entities'):
            print(f"\n🏷️  Most Mentioned Entities:")
            for entity in overview['top_entities'][:5]:
                print(f"   {entity['e.name']} ({entity['e.type']}) - mentioned {entity['e.frequency']} times")
        
        print("\n" + "=" * 50)
        print("💬 Interactive Query Mode")
        print("Commands:")
        print("  - Type any question or keywords to search")
        print("  - 'entity:<name>' to find sections mentioning an entity")
        print("  - 'section:<id>' to get section hierarchy")
        print("  - 'quit' to exit")
        print("-" * 50)
        
        while True:
            query = input("\n🔍 Query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            if query.startswith('entity:'):
                entity_name = query[7:].strip()
                results = querier.find_sections_by_entity(entity_name)
                print(format_search_results(results, f"Entity: {entity_name}"))
                
                # Also show related entities
                related = querier.find_related_entities(entity_name, 5)
                if related:
                    print(f"\n🔗 Related Entities:")
                    for rel in related:
                        print(f"   {rel['e2.name']} ({rel['e2.type']}) - co-occurs {rel['r.frequency']} times")
            
            elif query.startswith('section:'):
                section_id = query[8:].strip()
                hierarchy = querier.get_section_hierarchy(section_id)
                
                if hierarchy.get('section'):
                    s = hierarchy['section']
                    print(f"\n📑 Section {s['s.id']}: {s['s.title']}")
                    print(f"   Level: {s['s.level']}, Page: {s['s.page_number']}")
                    
                    if hierarchy.get('parent'):
                        p = hierarchy['parent']
                        print(f"   ⬆️  Parent: {p['parent.id']} - {p['parent.title']}")
                    
                    if hierarchy.get('children'):
                        print(f"   ⬇️  Children:")
                        for child in hierarchy['children']:
                            print(f"      {child['child.id']}: {child['child.title']}")
                    
                    # Show text preview
                    text = s.get('s.text', '')
                    if text:
                        preview = text[:300] + "..." if len(text) > 300 else text
                        print(f"   📝 Content: {preview}")
                else:
                    print(f"❌ Section '{section_id}' not found")
            
            else:
                # Semantic search
                results = querier.semantic_search_sections(query)
                print(format_search_results(results, query))
        
        querier.close()
        print("\n👋 Goodbye!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure Neo4j is running and the graph has been created.")

if __name__ == "__main__":
    main() 