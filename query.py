#!/usr/bin/env python3
"""
GraphRAG Query Pipeline - Query the Neo4j graph using semantic search and relationship traversal
"""

from neo4j import GraphDatabase
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
import spacy
from typing import List
from spacy.lang.en.stop_words import STOP_WORDS
import re

# Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from utils.custom_logger import get_logger
logging = get_logger()

class GraphRAGQuery:
    def __init__(self, uri: str = "bolt://localhost:7687", username: str = "neo4j", password: str = "password"):
        """Initialize Neo4j connection and OpenAI client."""
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        print("✓ Connected to Neo4j")
        
        # Initialize OpenAI client
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️ Warning: OPENAI_API_KEY not found in environment variables")
            self.openai_client = None
        else:
            self.openai_client = OpenAI(api_key=api_key)
            print("✓ Connected to OpenAI")
        
        # Initialize ML models as None
        self.nlp = None
        self.embedding_model = None
        self._models_initialized = False
    
    def _initialize_models(self):
        """Initialize ML models if they haven't been initialized yet."""
        if not self._models_initialized:
            print("🤖 Loading ML models...")
            self.nlp = spacy.load("en_core_web_trf")  # Transformer-based NER and noun chunks
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self._models_initialized = True
            print("✓ ML models loaded")
    
    def close(self):
        """Close Neo4j connection."""
        self.driver.close()
        print("✓ Connection closed")
    
        
    def clean_phrase(self, phrase: str) -> str:
        tokens = [t for t in phrase.split() if t.lower() not in STOP_WORDS]
        return " ".join(tokens)


    def fix_hyphenated_linebreaks(self, text: str) -> str:
        # Removes hyphen + newline (or hyphen + space + newline), joins word
        text = re.sub(r'-\s*\n\s*', '', text)
        # Optional: Replace newlines with space
        text = re.sub(r'\n+', ' ', text)
        return text

    def extract_keywords(self, text: str) -> List[str]:

        self._initialize_models()
        text = self.fix_hyphenated_linebreaks(text)
        doc = self.nlp(text)
        phrases = set()

        # 1. Cleaned noun chunks
        for chunk in doc.noun_chunks:
            phrase = self.clean_phrase(chunk.text.strip()).lower()
            if 1 <= len(phrase.split()) <= 5:
                phrases.add(phrase)

        # 2. Cleaned verb + object phrases
        for token in doc:
            if token.pos_ == "VERB":
                for child in token.children:
                    if child.dep_ in {"dobj", "attr", "pobj"}:
                        phrase = self.clean_phrase(f"{token.text} {child.text}")
                        if 1 <= len(phrase.split()) <= 5:
                            phrases.add(phrase)


        x = list(set(sorted(p for p in phrases if p and not all(w.lower() in STOP_WORDS for w in p.split()))))

        # lower case the keywords
        x = [keyword.lower() for keyword in x]
        return x        
    def get_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for the query text."""
        self._initialize_models()
        return self.embedding_model.encode(query, show_progress_bar=False)
    
    def semantic_search(self, query: str, limit: int = 5) -> List[Dict]:
        """Perform semantic search using query embedding."""

        self._initialize_models()

        query_embedding = self.get_query_embedding(query)
        
        with self.driver.session() as session:
            # Get all nodes with embeddings
            result = session.run("""
                MATCH (n)
                WHERE n.embedding IS NOT NULL
                RETURN n.id as id, n.title as title, n.text as text, n.embedding as embedding,
                       n.page_number as page_number, n.level as level, n.keywords as keywords
            """)
            
            # Calculate similarities in Python
            nodes = []
            for record in result:
                node_embedding = np.array(record['embedding'])
                similarity = cosine_similarity([query_embedding], [node_embedding])[0][0]
                # check the number of keywords in the query matching the number of keywords in the node
                # query_keywords = self.extract_keywords(query)
                # node_keywords = self.extract_keywords(record['text'])
                # keyword_match_count = len(set(query_keywords) & set(node_keywords))

                nodes.append({
                    'id': record['id'],
                    'title': record['title'],
                    'text': record['text'],
                    'similarity': similarity,
                    'cosine_similarity': similarity,
                    'page_number': record['page_number'],
                    'level': record['level'],
                    'keywords': record['keywords']
                    # 'keyword_match_count': keyword_match_count
                })
            
            # Sort by similarity and return top results
            nodes.sort(key=lambda x: x['similarity'], reverse=True)
            return nodes[:limit]
    
    def keyword_search(self, query: str, limit: int = 5) -> List[Dict]:
        """Search using extracted keywords."""
        keywords = self.extract_keywords(query)
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n)
                WHERE any(kw in $keywords WHERE toLower(n.title) CONTAINS toLower(kw))
                OR any(kw in $keywords WHERE toLower(n.text) CONTAINS toLower(kw))
                RETURN n.id as id, n.title as title, n.text as text
                LIMIT $limit
            """, keywords=keywords, limit=limit)
            
            return [dict(record) for record in result]
    
    def keyword_match_search(self, query: str, limit: int = 30, excluded_keywords: List[str] = None) -> List[Dict]:
        """Search nodes based on the number of keyword overlaps with the query (ties broken by cosine similarity)."""
        # Extract query keywords and query embedding once
        query_keywords = self.extract_keywords(query)
        query_embedding = self.get_query_embedding(query)

        with self.driver.session() as session:
            # Retrieve all nodes that have keywords and embeddings stored
            result = session.run("""
                MATCH (n)
                WHERE n.keywords IS NOT NULL AND n.embedding IS NOT NULL
                RETURN n.id as id, n.title as title, n.text as text, n.embedding as embedding,
                       n.page_number as page_number, n.level as level, n.keywords as keywords
            """)

            nodes = []
            for record in result:
                node_keywords = record["keywords"] or []

                # Calculate keyword matches between query and node
                matching_keywords, keyword_count = self._calculate_keyword_matches(
                    node_keywords,
                    query_keywords,
                    excluded_keywords
                )

                # Calculate cosine similarity for tie-breaking
                node_embedding = np.array(record["embedding"])
                similarity = cosine_similarity([query_embedding], [node_embedding])[0][0]

                nodes.append({
                    "id": record["id"],
                    "title": record["title"],
                    "text": record["text"],
                    "similarity": similarity,
                    "cosine_similarity": similarity,
                    "page_number": record["page_number"],
                    "level": record["level"],
                    "keywords": node_keywords,
                    "matching_keywords": matching_keywords,
                    "keyword_match_count": keyword_count
                })

            # Sort primarily by keyword match count, then by similarity
            nodes.sort(key=lambda x: (x["keyword_match_count"], x["cosine_similarity"]), reverse=True)
            return nodes[:limit], query_keywords
    
    def get_related_content(self, node_id: str, relationship_type: str, depth: int = 1) -> List[Dict]:
        """Get related content based on relationship type and depth."""
        with self.driver.session() as session:
            result = session.run(f"""
                MATCH path = (start)-[:{relationship_type}*1..{depth}]->(related)
                WHERE start.id = $node_id
                RETURN related.id as id, related.title as title, related.text as text,
                       related.page_number as page_number, related.level as level,
                       length(path) as depth
                ORDER BY depth
            """, node_id=node_id)
            
            return [dict(record) for record in result]
    
    def _calculate_keyword_matches(self, main_keywords: List[str], related_keywords: List[str], excluded_keywords: List[str] = None) -> tuple:
        """Calculate matching keywords and count between two nodes after filtering excluded keywords."""
        # Convert to lowercase sets
        main_keywords_set = set(kw.lower() for kw in main_keywords)
        related_keywords_set = set(kw.lower() for kw in related_keywords)
        
        # Remove excluded keywords from main node keywords
        if excluded_keywords:
            excluded_keywords_lower = set(kw.lower() for kw in excluded_keywords)
            main_keywords_set = main_keywords_set - excluded_keywords_lower
        
        # Find matching keywords
        matching_keywords = list(main_keywords_set & related_keywords_set)
        return matching_keywords, len(matching_keywords)

    def get_relationship_details(self, main_node_id: str, related_node_id: str, excluded_keywords: List[str] = None) -> List[Dict]:
        """Get details about the relationship between two nodes."""
        with self.driver.session() as session:
            # Get relationship details with direction
            result = session.run("""
                MATCH (n)-[r]-(m)
                WHERE n.id = $main_id AND m.id = $related_id
                RETURN type(r) as rel_type, r, 
                       CASE 
                           WHEN startNode(r) = n THEN 'outgoing'
                           ELSE 'incoming'
                       END as direction,
                       n.keywords as main_keywords,
                       m.keywords as related_keywords
            """, main_id=main_node_id, related_id=related_node_id)
            
            relationships = []
            for record in result:
                rel_type = record['rel_type']
                rel_props = dict(record['r'])
                direction = record['direction']
                
                # Calculate keyword matches for all relationship types
                matching_keywords, keyword_count = self._calculate_keyword_matches(
                    record['main_keywords'],
                    record['related_keywords'],
                    excluded_keywords
                )
                rel_props['matching_keywords'] = matching_keywords
                rel_props['keyword_match_count'] = keyword_count
                
                relationships.append({
                    'type': rel_type,
                    'properties': rel_props,
                    'direction': direction
                })
            return relationships

    def _load_prompt_template(self) -> str:
        """Load the prompt template from file."""
        try:
            template_path = Path(__file__).parent / "prompt_template.txt"
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print("⚠️ Warning: prompt_template.txt not found, using default template")
            return """You are a knowledgeable assistant. Answer the user's query based on the provided context.

            User Query: {user_query}

            Main Section: {main_node_title}
            Content: {main_node_content}

            Connected Sections:
            {relevant_nodes}

            Please provide a comprehensive answer based on this context."""

    def _format_relevant_nodes(self, context_nodes: List[Dict]) -> str:
        """Format the relevant nodes for the prompt."""
        if not context_nodes:
            return "No additional connected sections found."
        
        formatted_nodes = []
        for i, node in enumerate(context_nodes, 1):
            relationship_info = ""
            if 'cosine_similarity' in node:
                relationship_info += f"(Similarity: {node['cosine_similarity']:.2f}"
            if 'keyword_match_count' in node and node['keyword_match_count'] > 0:
                relationship_info += f", Keyword matches: {node['keyword_match_count']}"
            if relationship_info:
                relationship_info += ")"
            
            formatted_node = f"""
            **Section ID {node.get('id', 'N/A')} : {node.get('title', 'N/A')}** {relationship_info}
            - **Page:** {node.get('page_number', 'N/A')}
            - **Level:** {node.get('level', 'N/A')}
            - **Content:** {node.get('text', 'No content available')}
            """
            formatted_nodes.append(formatted_node)
        
        return "\n".join(formatted_nodes)

    def _construct_prompt(self, user_query: str, main_node: Dict, context_nodes: List[Dict]) -> str:
        """Construct the prompt using the template and provided data."""
        template = self._load_prompt_template()
        
        # Format relevant nodes
        relevant_nodes_text = self._format_relevant_nodes(context_nodes)
        
        # Fill in the template
        prompt = template.format(
            user_query=user_query,
            main_node_id=main_node.get('id', 'N/A'),
            main_node_title=main_node.get('title', 'N/A'),
            main_node_page=main_node.get('page_number', 'N/A'),
            main_node_level=main_node.get('level', 'N/A'),
            main_node_content=main_node.get('text', 'No content available'),
            relevant_nodes=relevant_nodes_text
        )
        
        return prompt

    def _call_openai_api(self, prompt: str, model: str = "gpt-4o-mini", max_tokens: int = 1500) -> str:
        """Call OpenAI API with the constructed prompt."""
        if not self.openai_client:
            return "OpenAI client not available. Please check your API key configuration."
        
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides comprehensive answers based on document content."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return f"Error generating LLM response: {str(e)}"

    def get_connected_nodes(self, query: str, main_min_keyword_matches: int = 2, main_similarity_threshold: float = 0.95,
                          connected_min_keyword_matches: int = 2, connected_similarity_threshold: float = 0.95,
                          filter_type: str = "Keyword Matches", max_results: int = 5, excluded_keywords: List[str] = None) -> Dict:
        """Get connected nodes based on semantic search and relationship traversal."""
        
        logging.info("--------------------------------")
        print("Request arguments : ")
        logging.info("User Query : " + query)
        logging.info("main_min_keyword_matches : " + str(main_min_keyword_matches) + " main_similarity_threshold : " + str(main_similarity_threshold) + " connected_min_keyword_matches : " + str(connected_min_keyword_matches) + " connected_similarity_threshold : " + str(connected_similarity_threshold) + " filter_type : " + filter_type + " max_results : " + str(max_results) + " excluded_keywords : " + str(excluded_keywords))
        logging.info("--------------------------------")
        print("Query Recieved : " + query)
        
        # Select main node candidates based on filter type
        if filter_type == "Keyword Matches":
            candidate_results, query_keywords = self.keyword_match_search(query, limit=30, excluded_keywords=excluded_keywords)
        else:  # "Similarity Score" or any other value defaults to similarity search
            candidate_results = self.semantic_search(query, limit=30)

        if not candidate_results:
            return {"status": "error", "answer": "No content found for the given query. Try lowering the threshold values.", "context": []}

        for i in range(len(candidate_results)):
            logging.info("candidate_results : " + str(candidate_results[i]['id']) + " --> " + str(candidate_results[i]['keyword_match_count']) + " --> " + str(candidate_results[i]['cosine_similarity']) + " --> " + str(candidate_results[i]['matching_keywords']))
        
        main_node = candidate_results[0]

        # Verify main node against appropriate threshold
        if filter_type == "Keyword Matches":
            if main_node.get("keyword_match_count", 0) < main_min_keyword_matches:
                return {"status": "error", "answer": "No content found matching the main node keyword threshold. Try lowering the threshold values.", "context": []}
        else:  # Similarity threshold check
            if main_node["similarity"] < main_similarity_threshold:
                return {"status": "error", "answer": "No content found matching the main node similarity threshold. Try lowering the threshold values.", "context": []}

        # get the query keywords
        query_keywords = self.extract_keywords(query)
        logging.info("query_keywords : " + str(query_keywords))

        # Get main node keywords from semantic search results
        main_node_keywords = main_node.get('keywords', [])
        # print("main_node_keywords : " , main_node_keywords)
        
        # Calculate matching keywords
        matching_keywords, keyword_count = self._calculate_keyword_matches(
            main_node_keywords,
            list(query_keywords),  # Convert set to list for consistency
            excluded_keywords
        )
        # print("matching_keywords : " , matching_keywords)
        # print("keyword_count : " , keyword_count)
        main_node['matching_keywords'] = matching_keywords
        main_node['keyword_match_count'] = keyword_count
        
        # Initialize categorized context dictionary
        categorized_context = {
            'parent': [],
            'subsections': [],
            'next': [],
            'keyword_matches': [],
            'semantic_matches': []
        }
        
        # Track seen nodes to avoid duplicates
        seen_ids = {main_node['id']}

        # 1. Get parent content (always include if exists)
        parent_content = self.get_related_content(main_node['id'], 'PARENT', depth=1)
        for item in parent_content:
            if item['id'] not in seen_ids:
                seen_ids.add(item['id'])
                categorized_context['parent'].append(item)
        
        # print("parent_content : " , parent_content)
        # 2. Get subsection content (apply thresholds)
        subsection_content = self.get_related_content(main_node['id'], 'HAS_SUBSECTION', depth=1)
        for item in subsection_content:
            if item['id'] not in seen_ids:
                relationships = self.get_relationship_details(main_node['id'], item['id'], excluded_keywords)
                should_include = False
                
                for rel in relationships:
                    if rel['type'] == 'HAS_SUBSECTION':
                        # Use our calculated keyword count
                        keyword_count = rel['properties']['keyword_match_count']
                        if keyword_count >= connected_min_keyword_matches:
                            # Check similarity
                            similarity = rel['properties'].get('similarity_score', 0)
                            if similarity >= connected_similarity_threshold:
                                should_include = True
                                break
                
                if should_include:
                    seen_ids.add(item['id'])
                    categorized_context['subsections'].append(item)
        
        # print("subsection_content : " , subsection_content)
        # 3. Get next content (apply thresholds)
        next_content = self.get_related_content(main_node['id'], 'NEXT', depth=1)
        for item in next_content:
            if item['id'] not in seen_ids:
                relationships = self.get_relationship_details(main_node['id'], item['id'], excluded_keywords)
                should_include = False
                
                for rel in relationships:
                    if rel['type'] == 'NEXT':
                        # Use our calculated keyword count
                        keyword_count = rel['properties']['keyword_match_count']
                        if keyword_count >= connected_min_keyword_matches:
                            # Check similarity
                            similarity = rel['properties'].get('similarity_score', 0)
                            if similarity >= connected_similarity_threshold:
                                should_include = True
                                break
                
                if should_include:
                    seen_ids.add(item['id'])
                    categorized_context['next'].append(item)
        # print("next_content : " , next_content)
        # 4. Get keyword mention content (apply keyword threshold)
        mention_content = self.get_related_content(main_node['id'], 'KEYWORD_MENTIONS', depth=1)
        for item in mention_content:
            if item['id'] not in seen_ids:
                relationships = self.get_relationship_details(main_node['id'], item['id'], excluded_keywords)
                for rel in relationships:
                    if rel['type'] == 'KEYWORD_MENTIONS':
                        # Use our calculated keyword count
                        keyword_count = rel['properties']['keyword_match_count']
                        if keyword_count >= connected_min_keyword_matches:
                            seen_ids.add(item['id'])
                            categorized_context['keyword_matches'].append(item)
                            break
        # print("mention_content : " , mention_content)
        # 5. Get similar content (apply similarity threshold)
        similar_content = self.get_related_content(main_node['id'], 'SEMANTIC_SIMILAR_TO', depth=1)
        for item in similar_content:
            if item['id'] not in seen_ids:
                relationships = self.get_relationship_details(main_node['id'], item['id'], excluded_keywords)
                for rel in relationships:
                    if rel['type'] == 'SEMANTIC_SIMILAR_TO':
                        similarity = rel['properties'].get('similarity_score', 0)
                        if similarity >= connected_similarity_threshold:
                            seen_ids.add(item['id'])
                            categorized_context['semantic_matches'].append(item)
                            break
        # print("similar_content : " , similar_content)
        # Calculate cosine similarity and keyword matches for sorting within categories
        main_embedding = self.get_query_embedding(main_node['text'])
        for category in categorized_context.values():
            for item in category:
                # Calculate cosine similarity
                item_embedding = self.get_query_embedding(item['text'])
                similarity = cosine_similarity([main_embedding], [item_embedding])[0][0]
                item['cosine_similarity'] = float(similarity)
                # Get keyword match count
                relationships = self.get_relationship_details(main_node['id'], item['id'], excluded_keywords)
                max_keyword_matches = 0
                matching_keywords = []
                for rel in relationships:
                    if rel['type'] == 'KEYWORD_MENTIONS':
                        keyword_count = rel['properties']['keyword_match_count']
                        if keyword_count > max_keyword_matches:
                            max_keyword_matches = keyword_count
                            matching_keywords = rel['properties']['matching_keywords']
                item['keyword_match_count'] = max_keyword_matches
                item['matching_keywords'] = matching_keywords
        
        # Sort within each category based on filter_type
        for category in categorized_context.values():
            if filter_type == "Similarity Score":
                category.sort(key=lambda x: x['cosine_similarity'], reverse=True)
            else:
                category.sort(key=lambda x: (x['keyword_match_count'], x['cosine_similarity']), reverse=True)
        
        # Combine all categories in the specified order
        all_context = []
        for category in ['parent', 'subsections', 'next', 'keyword_matches', 'semantic_matches']:
            all_context.extend(categorized_context[category])
        
        # Apply max_results limit to the combined list
        all_context = all_context[:max_results]
        
        # Create visualization query
        all_node_ids = [main_node['id']] + [item['id'] for item in all_context]
        visualization_query = f"""
        MATCH (n)
        WHERE n.id IN {all_node_ids}
        WITH n
        MATCH (n)-[r]-(m)
        WHERE m.id IN {all_node_ids}
        RETURN n, r, m
        """
        
        # Generate LLM response using OpenAI
        llm_response = None
        if self.openai_client:
            try:
                prompt = self._construct_prompt(query, main_node, all_context)
                llm_response = self._call_openai_api(prompt)
                print("✓ Generated LLM response")
            except Exception as e:
                print(f"Error generating LLM response: {e}")
                llm_response = f"Error generating LLM response: {str(e)}"
        else:
            llm_response = "LLM response not available - OpenAI client not configured"
        
        print("Status : Success")
        print("Primary Node : " + main_node.get('id', 'N/A'))

        print("--------------------------------")

        return {
            "status": "success",
            "id": main_node['id'],
            "section_number": main_node.get('id', 'N/A'),  # Using id as section number
            "answer": main_node['text'],
            "title": main_node['title'],
            "similarity_score": main_node['similarity'],
            "cosine_similarity": main_node['cosine_similarity'],
            "page_number": main_node.get('page_number', 'N/A'),
            "level": main_node.get('level', 'N/A'),
            "matching_keywords": main_node['matching_keywords'],
            "keyword_match_count": main_node['keyword_match_count'],
            "context": all_context,
            "visualization_query": visualization_query,
            "llm_response": llm_response,
            "prompt_used": self._construct_prompt(query, main_node, all_context) if self.openai_client else None
        }

if __name__ == "__main__":
    pass
