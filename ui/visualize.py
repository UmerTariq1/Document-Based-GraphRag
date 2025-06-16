# streamlit run ui/visualize.py
 
import streamlit as st
import sys
from pathlib import Path
import tempfile
import json
from typing import List, Dict

# Add parent directory to path to import query.py
sys.path.append(str(Path(__file__).parent.parent))
from query import GraphRAGQuery
from ingest import GraphRAGIngestion
from pdf_reader import parse_pdf

def initialize_session_state():
    """Initialize session state variables."""
    if 'query_engine' not in st.session_state:
        try:
            st.session_state.query_engine = GraphRAGQuery()
        except Exception as e:
            st.error(f"Error initializing query engine: {e}")
            st.stop()
    if 'last_query' not in st.session_state:
        st.session_state.last_query = None
    if 'last_results' not in st.session_state:
        st.session_state.last_results = None
    if 'excluded_keywords' not in st.session_state:
        st.session_state.excluded_keywords = set()

def convert_document_to_dict_list(document) -> List[Dict]:
    """Convert Document object to list of dictionaries for ingestion."""
    sections_data = []
    for section in document.sections:
        section_dict = {
            'id': section.id,
            'title': section.title,
            'text': section.text,
            'level': section.level,
            'parent_id': section.parent_id,
            'page_number': section.page_number,
            'is_toc_entry': section.is_toc_entry
        }
        sections_data.append(section_dict)
    return sections_data

def process_pdf_and_ingest(uploaded_file):
    """Process uploaded PDF and ingest into Neo4j in one seamless flow."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Save uploaded file temporarily
        status_text.text("📁 Processing uploaded PDF...")
        progress_bar.progress(10)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Step 2: Parse PDF to structured data
        status_text.text("🔍 Extracting content from PDF...")
        progress_bar.progress(30)
        
        document = parse_pdf(tmp_file_path)
        sections_data = convert_document_to_dict_list(document)
        
        if not sections_data:
            st.error("❌ No sections found in the PDF. Please check if the PDF has the expected structure.")
            return False
        
        # Step 3: Initialize ingestion engine
        status_text.text("🔗 Connecting to Neo4j database...")
        progress_bar.progress(50)
        
        ingestion = GraphRAGIngestion()
        
        # Step 4: Clear existing data and ingest new data
        status_text.text("🧹 Clearing existing data...")
        progress_bar.progress(60)
        ingestion.clear_database()
        
        # Step 5: Create graph structure
        status_text.text("🏗️ Creating section nodes...")
        progress_bar.progress(70)
        ingestion.create_section_nodes(sections_data)
        
        status_text.text("🔗 Creating relationships...")
        progress_bar.progress(80)
        ingestion.create_hierarchical_relationships(sections_data)
        ingestion.create_sibling_relationships(sections_data)
        ingestion.create_mention_relationships(sections_data)
        ingestion.create_similarity_relationships(sections_data)
        
        status_text.text("🔍 Creating indexes...")
        progress_bar.progress(90)
        ingestion.create_indexes()
        
        # Step 6: Show completion
        progress_bar.progress(100)
        status_text.text("✅ Ingestion completed successfully!")
        
        # Clean up
        ingestion.close()
        Path(tmp_file_path).unlink()  # Delete temporary file
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error during ingestion: {e}")
        status_text.text("❌ Ingestion failed!")
        if 'ingestion' in locals():
            ingestion.close()
        if 'tmp_file_path' in locals():
            try:
                Path(tmp_file_path).unlink()
            except:
                pass
        return False

def show_ingestion_tab():
    """Display the ingestion tab content."""
    # Sidebar content for ingestion
    with st.sidebar:
        st.markdown("### 📚 Document-Based GraphRAG")
        st.markdown("""
        **Transform your PDFs into intelligent knowledge graphs!**
        
        This tool helps you:
        - 📄 **Extract** structured content from PDF documents
        - 🕸️ **Build** semantic knowledge graphs with AI
        - 🔍 **Query** your documents using natural language
        - 🧠 **Get** AI-powered insights and answers
        
        **Perfect for:**
        - Technical documentation
        - Research papers  
        - Product manuals
        - Policy documents
        - Educational materials
        
        **How it works:**
        1. Upload your PDF
        2. AI extracts sections and relationships
        3. Query in plain English
        4. Get comprehensive answers!
        """)
        
        st.markdown("---")
        st.markdown("**💡 Tip:** Start with technical documents that have clear section structures for best results.")
    
    # Main content area
    st.markdown("## 🚀 Document Ingestion")
    st.markdown("""
    Upload a PDF document to extract its content and build an intelligent knowledge graph.
    The system will automatically identify sections, extract relationships, and prepare your document for intelligent querying.
    """)
    
    # File upload section
    st.markdown("### 📁 Upload PDF Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF document with structured content (sections, subsections, etc.)"
    )
    
    if uploaded_file is not None:
        # Show file details
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.2f} KB"
        }
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**File Details:**")
            for key, value in file_details.items():
                st.text(f"{key}: {value}")
        
        with col2:
            st.markdown("**What happens next:**")
            st.text("✓ Extract document structure")
            st.text("✓ Identify sections and relationships") 
            st.text("✓ Generate semantic embeddings")
            st.text("✓ Build knowledge graph in Neo4j")
        
        st.markdown("---")
        
        # Ingestion button and process
        if st.button("🚀 Start Ingestion", type="primary", use_container_width=True):
            st.markdown("### 📊 Ingestion Progress")
            
            success = process_pdf_and_ingest(uploaded_file)
            
            if success:
                st.success("🎉 **Ingestion completed successfully!**")
                st.markdown("""
                Your document has been processed and is now ready for querying:
                - ✅ Content extracted and structured
                - ✅ Knowledge graph created in Neo4j
                - ✅ Semantic relationships established
                - ✅ Ready for intelligent querying
                
                **Next steps:** Switch to the **Query** tab to start asking questions about your document!
                """)
                
                # Show some basic stats
                if hasattr(st.session_state, 'query_engine'):
                    try:
                        # Refresh the query engine connection
                        st.session_state.query_engine = GraphRAGQuery()
                        st.balloons()
                    except:
                        pass
            else:
                st.error("❌ Ingestion failed. Please check the error messages above and try again.")
    
    else:
        st.info("👆 Please upload a PDF file to begin the ingestion process.")
        
        # Show example of what kind of documents work well
        with st.expander("💡 What kind of documents work best?"):
            st.markdown("""
            **Ideal documents:**
            - Technical manuals with numbered sections (1.1, 1.2, etc.)
            - Research papers with clear structure
            - Policy documents with hierarchical organization
            - Educational materials with chapters and sections
            
            **Document requirements:**
            - PDF format
            - Text-based content (not scanned images)
            - Clear section numbering and titles
            - Structured layout with headings
            
            **Example structure:**
            ```
            1. Introduction
            1.1 Overview
            1.2 Purpose
            2. Main Content
            2.1 Features
            2.2 Benefits
            ```
            """)

def get_relationship_details(query_engine, main_node_id: str, related_node_id: str) -> dict:
    """Get details about the relationship between two nodes."""
    with query_engine.driver.session() as session:
        # Get relationship details with direction
        result = session.run("""
            MATCH (n)-[r]-(m)
            WHERE n.id = $main_id AND m.id = $related_id
            RETURN type(r) as rel_type, r, 
                   CASE 
                       WHEN startNode(r) = n THEN 'outgoing'
                       ELSE 'incoming'
                   END as direction
        """, main_id=main_node_id, related_id=related_node_id)
        
        relationships = []
        for record in result:
            rel_type = record['rel_type']
            rel_props = dict(record['r'])
            direction = record['direction']
            
            # For KEYWORD_MENTIONS, get the actual matching keywords
            if rel_type == 'KEYWORD_MENTIONS':
                # Get keywords from both nodes
                keywords_result = session.run("""
                    MATCH (n), (m)
                    WHERE n.id = $main_id AND m.id = $related_id
                    RETURN n.keywords as main_keywords, m.keywords as related_keywords
                """, main_id=main_node_id, related_id=related_node_id)
                
                keywords_record = keywords_result.single()
                if keywords_record:
                    main_keywords = set(kw.lower() for kw in keywords_record['main_keywords'])
                    related_keywords = set(kw.lower() for kw in keywords_record['related_keywords'])
                    matching_keywords = main_keywords & related_keywords
                    rel_props['matching_keywords'] = list(matching_keywords)
            
            relationships.append({
                'type': rel_type,
                'properties': rel_props,
                'direction': direction
            })
        return relationships

def display_results(results, query_engine):
    """Display query results in a structured way."""
    # LLM Response section (show first if available)
    if results.get('llm_response'):
        st.markdown("## 🤖 AI-Generated Response")
        with st.container():
            st.markdown(results['llm_response'])
        
        # Add expandable section for prompt debugging
        if results.get('prompt_used'):
            with st.expander("🔍 View Prompt Used (Debug)"):
                st.text(results['prompt_used'])
        
        st.markdown("---")
    
    # Main answer section
    with st.container():
        # Get keyword matches for main node
        main_keyword_matches = results.get('keyword_match_count', 0)
        main_matching_keywords = results.get('matching_keywords', [])

        # Make section number and title as heading
        st.markdown(f"### 📄 Source: Section {results['id']}: {results['title']}")
        
        st.markdown(f"**Page Number :** {results.get('page_number', 'N/A')} , **Query Similarity:** {results['similarity_score']:.2f}, **Number of matching keywords:** {main_keyword_matches}")

        if main_keyword_matches > 0:
            st.markdown(f"**Matching Keywords:** {', '.join(main_matching_keywords)}")
        else:
            st.markdown(f"**Matching Keywords:** _No matching keywords found._")

        st.markdown("**Original Content:**")
        if results['answer']:
            st.markdown(results['answer'])
        else:
            st.markdown("_This section has no body of text._")
        
        # Related context section
        st.markdown("### 📚 Connected Nodes")
        for item in results['context']:
            try:
                relationships = get_relationship_details(query_engine, results['id'], item['id'])
                rel_descriptions = []
                seen_details = set()  # Track seen relationship details to avoid duplicates
                
                # Always show similarity score with main node
                similarity_desc = f"🔄 Similarity with main answer: {item['cosine_similarity']:.2f}"
                rel_descriptions.append(similarity_desc)
                
                # Get keyword matches for this node
                keyword_matches = item.get('keyword_match_count', 0)
                matching_keywords = item.get('matching_keywords', [])
                
                # Add keyword match information
                if keyword_matches > 0:
                    keyword_desc = f"🔑 Keyword matches: {keyword_matches}"
                    rel_descriptions.append(keyword_desc)
                    if matching_keywords:
                        rel_descriptions.append(f"🔑 Matching keywords: {', '.join(matching_keywords)}")
                else:
                    rel_descriptions.append("_No matching keywords found._")
                
                for rel in relationships:
                    rel_type = rel['type']
                    props = rel['properties']
                    direction = rel['direction']
                    
                    # Only show relationship if it's from the main node to the related node
                    if direction == 'outgoing':
                        if rel_type == 'SEMANTIC_SIMILAR_TO':
                            similarity = props.get('similarity_score', 0)
                            similarity_desc = f"🔄 Semantic similarity: {similarity:.2f}"
                            if similarity_desc not in seen_details:
                                rel_descriptions.append(similarity_desc)
                                seen_details.add(similarity_desc)
                        elif rel_type == 'HAS_SUBSECTION':
                            subsection_desc = "⬇️ This is a subsection of the main answer"
                            if subsection_desc not in seen_details:
                                rel_descriptions.append(subsection_desc)
                                seen_details.add(subsection_desc)
                        elif rel_type == 'NEXT':
                            next_desc = "➡️ This section comes after the main answer"
                            if next_desc not in seen_details:
                                rel_descriptions.append(next_desc)
                                seen_details.add(next_desc)
                        elif rel_type == 'PARENT':
                            parent_desc = "⬆️ This is the parent section of the main answer"
                            if parent_desc not in seen_details:
                                rel_descriptions.append(parent_desc)
                                seen_details.add(parent_desc)
                
                with st.expander(f"📄 {item['title']} (Section Number: {item['id']}, Page: {item.get('page_number', 'N/A')})"):
                    if rel_descriptions:
                        st.markdown("**Relationship details:**")
                        for desc in rel_descriptions:
                            st.markdown(f"- {desc}")
                    st.markdown("**Content:**")
                    if item['text']:
                        st.markdown(item['text'])
                    else:
                        st.markdown("_This section has no body of text._")
            except Exception as e:
                st.error(f"Error displaying relationship details: {e}")
                # Fallback to basic display
                with st.expander(f"📄 {item['title']} (Section Number: {item['id']})"):
                    st.markdown("**Content:**")
                    if item['text']:
                        st.markdown(item['text'])
                    else:
                        st.markdown("_This section has no body of text._")
        
        # Visualization section
        st.markdown("### 🔍 Graph Visualization")
        st.markdown("Run this query in Neo4j Browser to visualize the graph:")
        st.code(results['visualization_query'], language="cypher")

def show_query_tab():
    """Display the query tab content."""
    # Sidebar with controls
    with st.sidebar:
        st.markdown("### ⚙️ Query Settings")
        
        # Main node thresholds
        st.markdown("**Main Node Thresholds:**")
        main_min_keyword_matches = st.slider(
            "Main Node - Minimum Keyword Matches",
            min_value=1,
            max_value=10,
            value=2,
            help="Minimum number of matching keywords required for the main node"
        )
        
        main_similarity_threshold = st.slider(
            "Main Node - Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.01,
            help="Minimum similarity score required for the main node"
        )
        
        # Result filtering
        st.markdown("### 🔍 Result Filtering")
        st.markdown("**Connected Nodes Thresholds:**")
        connected_min_keyword_matches = st.slider(
            "Connected Nodes - Minimum Keyword Matches",
            min_value=1,
            max_value=10,
            value=2,
            help="Minimum number of matching keywords required for connected nodes"
        )
        
        connected_similarity_threshold = st.slider(
            "Connected Nodes - Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.01,
            help="Minimum similarity score required for connected nodes"
        )
        
        filter_type = st.radio(
            "Sort by",
            ["Keyword Matches", "Similarity Score"],
            help="Choose how to sort the results"
        )
        
        max_results = st.slider(
            "Maximum Results",
            min_value=1,
            max_value=20,
            value=5,
            help="Maximum number of results to display"
        )
        
        # LLM Configuration
        st.markdown("### 🤖 LLM Settings")
        enable_llm = st.checkbox(
            "Enable AI Response",
            value=True,
            help="Generate AI-powered responses using OpenAI"
        )
        
        if enable_llm:
            llm_model = st.selectbox(
                "Model",
                ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
                index=0,
                help="Choose the OpenAI model to use"
            )
            
            max_tokens = st.slider(
                "Max Response Length",
                min_value=500,
                max_value=3000,
                value=1500,
                step=100,
                help="Maximum number of tokens in the AI response"
            )
        
        # Keyword exclusion
        st.markdown("### 🚫 Exclude Keywords")
        exclude_keyword = st.text_input(
            "Add keyword to exclude",
            placeholder="Enter keyword or phrase to exclude",
            help="Keywords entered here will be ignored during keyword matching"
        )
        
        # Add keyword to exclusion list
        if exclude_keyword and st.button("Add"):
            st.session_state.excluded_keywords.add(exclude_keyword.lower())
            st.rerun()
        
        # Display excluded keywords with remove buttons
        if st.session_state.excluded_keywords:
            st.markdown("**Excluded keywords:**")
            cols = st.columns(2)  # Create 3 columns for the grid
            for i, keyword in enumerate(sorted(st.session_state.excluded_keywords)):
                col_idx = i % 2
                with cols[col_idx]:
                    if st.button(f"❌ {keyword}", key=f"remove_{keyword}"):
                        st.session_state.excluded_keywords.remove(keyword)
                        st.rerun()
        
        if st.button("Clear Results"):
            st.session_state.last_query = None
            st.session_state.last_results = None
            st.rerun()
    
    # Main query interface
    st.markdown("## 🔍 Query Your Knowledge Graph")
    st.markdown("""
    Ask questions about your ingested documents in natural language.
    The system will find relevant content and show how different pieces of information are connected.
    """)
    
    # Test button and query input in a row
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🧪 Test Query", help="Click to run a test query"):
            st.session_state.query_input = "How can learning content be offered, and which international standards (such as SCORM, xAPI, LTI, and QTI) are supported?"
            st.rerun()
    
    with col2:
        query = st.text_input(
            "Enter your query:",
            placeholder="e.g., What is the imc Learning Suite?",
            key="query_input"
        )
    
    # Search button
    if st.button("🔍 Search", type="primary"):
        if query:
            with st.spinner("Searching and generating AI response..."):
                try:
                    # Temporarily modify the query engine for this search if LLM is disabled
                    if not enable_llm:
                        # Store original client
                        original_client = st.session_state.query_engine.openai_client
                        st.session_state.query_engine.openai_client = None
                    
                    results = st.session_state.query_engine.get_connected_nodes(
                        query,
                        main_min_keyword_matches=main_min_keyword_matches,
                        main_similarity_threshold=main_similarity_threshold,
                        connected_min_keyword_matches=connected_min_keyword_matches,
                        connected_similarity_threshold=connected_similarity_threshold,
                        filter_type=filter_type,
                        max_results=max_results,
                        excluded_keywords=list(st.session_state.excluded_keywords)
                    )
                    
                    # If LLM is enabled and we have a client, regenerate response with custom settings
                    if enable_llm and st.session_state.query_engine.openai_client and results.get('status') == 'success':
                        try:
                            # Get the main node and context from results
                            main_node = {
                                'id': results['id'],
                                'title': results['title'],
                                'text': results['answer'],
                                'page_number': results.get('page_number'),
                                'level': results.get('level')
                            }
                            
                            # Regenerate LLM response with custom settings
                            prompt = st.session_state.query_engine._construct_prompt(query, main_node, results['context'])
                            llm_response = st.session_state.query_engine._call_openai_api(prompt, model=llm_model, max_tokens=max_tokens)
                            results['llm_response'] = llm_response
                            results['prompt_used'] = prompt
                        except Exception as llm_error:
                            results['llm_response'] = f"Error generating custom LLM response: {str(llm_error)}"
                    
                    # Restore original client if it was temporarily disabled
                    if not enable_llm:
                        st.session_state.query_engine.openai_client = original_client
                    
                    st.session_state.last_query = query
                    st.session_state.last_results = results
                except Exception as e:
                    st.error(f"Error during query: {e}")
                    st.error("Please check if Neo4j is running and accessible.")
        else:
            st.warning("Please enter a query.")
    
    # Display results if available
    if st.session_state.last_results:
        st.markdown("---")
        if st.session_state.last_results.get('status') == 'error':
            st.error(st.session_state.last_results['answer'])
        else:
            display_results(st.session_state.last_results, st.session_state.query_engine)

def main():
    st.set_page_config(
        page_title="GraphRAG - Document Intelligence",
        page_icon="🔍",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("🔍 Document-Based GraphRAG")
    st.markdown("""
    Transform your PDF documents into intelligent knowledge graphs and query them using natural language.
    Upload documents, build semantic relationships, and get AI-powered insights.
    """)
    
    # Create tabs
    tab1, tab2 = st.tabs(["📁 Ingestion", "🔍 Query"])
    
    with tab1:
        show_ingestion_tab()
    
    with tab2:
        show_query_tab()

if __name__ == "__main__":
    main()
