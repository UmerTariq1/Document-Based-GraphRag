# streamlit run ui/visualize.py


import logging
logging.getLogger("streamlit").setLevel(logging.ERROR)

import sys, os
import types

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.custom_logger import get_logger
from utils.neo4j_utils import get_graph_stats

logging = get_logger()

# Patch torch.classes to prevent Streamlit from introspecting its __path__
try:
    import torch

    class DummyPath:
        # Fake path object to avoid __path__._path lookup
        def __getattr__(self, name):
            return []

    if isinstance(torch.classes, types.ModuleType):
        torch.classes.__path__ = DummyPath()
except Exception:
    pass  # Safe fallback if torch is not installed

import streamlit as st
from pathlib import Path
import tempfile
from typing import List, Dict

# Add parent directory to path to import query.py
sys.path.append(str(Path(__file__).parent.parent))
from query import GraphRAGQuery
from ingest import GraphRAGIngestion
from pdf_reader import parse_pdf

def load_css():
    """Load and inject custom CSS styles."""
    css_file = Path(__file__).parent / "style.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning("CSS file not found. Using default styling.")

def styled_container(css_class: str, content_func):
    """Create a styled container using Streamlit's container and CSS classes."""
    container = st.container()
    with container:
        # Apply CSS class to container
        container._get_widget_id = lambda: f"styled-{css_class}"
        st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
        content_func()
        st.markdown('</div>', unsafe_allow_html=True)

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
    # Interactive candidate selection state
    if 'candidate_list' not in st.session_state:
        st.session_state.candidate_list = None
    if 'awaiting_candidate_selection' not in st.session_state:
        st.session_state.awaiting_candidate_selection = False
    if 'selected_candidate_id' not in st.session_state:
        st.session_state.selected_candidate_id = None

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

# --------------------------------------------------
# Utility: Fetch simple graph statistics (node & relationship counts)
# --------------------------------------------------

def process_pdf_and_ingest(uploaded_file):
    """Process uploaded PDF and ingest into Neo4j in one seamless flow."""
    # Create progress container with styling
    progress_container = st.container()
    with progress_container:
        st.markdown('<div class="progress-container slide-in">', unsafe_allow_html=True)
        progress_bar = st.progress(0, text="0%")
        status_text = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
    
    try:
        # Step 1: Save uploaded file temporarily
        status_text.text("📁 Processing uploaded PDF...")
        progress_bar.progress(10, text="10%")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Step 2: Parse PDF to structured data
        status_text.text("🔍 Extracting content from PDF...")
        progress_bar.progress(30, text="30%")
        
        document = parse_pdf(tmp_file_path)
        sections_data = convert_document_to_dict_list(document)
        
        if not sections_data:
            st.error("❌ No sections found in the PDF. Please check if the PDF has the expected structure.")
            return False
        
        # Step 3: Initialize ingestion engine
        status_text.text("🔗 Connecting to Neo4j database...")
        progress_bar.progress(50, text="50%")
        
        ingestion = GraphRAGIngestion()
        
        # Step 4: Clear existing data and ingest new data
        status_text.text("🧹 Clearing existing data...")
        progress_bar.progress(60, text="60%")
        ingestion.clear_database()
        
        # Step 5: Create graph structure
        status_text.text("🏗️ Creating section nodes... (might take a while)")
        progress_bar.progress(70, text="70%")
        ingestion.create_section_nodes(sections_data)

        status_text.text("🔗 Creating relationships...")
        progress_bar.progress(80, text="80%")
        ingestion.create_hierarchical_relationships(sections_data)
        ingestion.create_sibling_relationships(sections_data)
        ingestion.create_mention_relationships(sections_data)
        ingestion.create_similarity_relationships(sections_data)

        
        status_text.text("🔍 Creating indexes...")
        progress_bar.progress(90, text="90%")
        ingestion.create_indexes()
        
        # Step 6: Show completion
        progress_bar.progress(100, text="100%")
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
        # Create sidebar content in a container
        sidebar_container = st.container()
        with sidebar_container:
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
    
    # Main content area with proper styling
    main_container = st.container()
    with main_container:
        st.markdown("""
        <div class="ingestion-container slide-in">
        <h2>🚀 Document Ingestion</h2>
        <p>Upload a PDF document to extract its content and build an intelligent knowledge graph.
        The system will automatically identify sections, extract relationships, and prepare your document for intelligent querying.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # --------------------------------------------------
    # Graph status card - show existing data counts
    # --------------------------------------------------
    status_container = st.container()
    with status_container:
        st.markdown('<div class="glass-container slide-in">', unsafe_allow_html=True)
        # Attempt to get counts if query engine is available
        node_count, rel_count = 0, 0
        try:
            if 'query_engine' in st.session_state and st.session_state.query_engine:
                node_count, rel_count = get_graph_stats(st.session_state.query_engine.driver)
        except Exception as neo_err:
            # Silently ignore; counts remain zero
            print(f"Neo4j stats error: {neo_err}")
        if node_count > 0:
            st.markdown(f"✅ **Data already ingested.** \n\n📦 **{node_count} nodes** and 🔗 **{rel_count} relationships** currently in the graph.")
        else:
            st.markdown("⚠️ **No data found in the database. Please upload a PDF to begin ingestion.**")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # File upload section
    upload_container = st.container()
    with upload_container:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.markdown("### 📁 Upload PDF Document")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF document with structured content (sections, subsections, etc.)"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Show file details
        details_container = st.container()
        with details_container:
            st.markdown('<div class="glass-container">', unsafe_allow_html=True)
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
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Ingestion button and process
        if st.button("🚀 Start Ingestion", type="primary", use_container_width=True):
            st.markdown("### 📊 Ingestion Progress")
            
            success = process_pdf_and_ingest(uploaded_file)
            
            if success:
                success_container = st.container()
                with success_container:
                    st.markdown("""
                    <div class="success-container slide-in">
                    <h3>🎉 Ingestion completed successfully!</h3>
                    <p>Your document has been processed and is now ready for querying:</p>
                    <ul>
                        <li>✅ Content extracted and structured</li>
                        <li>✅ Knowledge graph created in Neo4j</li>
                        <li>✅ Semantic relationships established</li>
                        <li>✅ Ready for intelligent querying</li>
                    </ul>
                    <p><strong>Next steps:</strong> Switch to the <strong>Query</strong> tab to start asking questions about your document!</p>
                    </div>
                    """, unsafe_allow_html=True)
                
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
            <div class="glass-container">
            <h4>Ideal documents:</h4>
            <ul>
                <li>Technical manuals with numbered sections (1.1, 1.2, etc.)</li>
                <li>Research papers with clear structure</li>
                <li>Policy documents with hierarchical organization</li>
                <li>Educational materials with chapters and sections</li>
            </ul>
            <h4>Document requirements:</h4>
            <ul>
                <li>PDF format</li>
                <li>Text-based content (not scanned images)</li>
                <li>Clear section numbering and titles</li>
                <li>Structured layout with headings</li>
            </ul>
            <h4>Example structure:</h4>
                <ul>
                    1. Introduction
                    1.1 Overview
                    1.2 Purpose
                    2. Main Content
                    2.1 Features
                    2.2 Benefits
                </ul>
            </div>
            """, unsafe_allow_html=True)

def display_results(results, query_engine):
    """Display query results in a structured way with custom styling."""
    # Wrap entire results in answer wrapper
    results_container = st.container()
    with results_container:        
        # LLM Response section (show first if available)
        if results.get('llm_response'):
            st.markdown(f"""
            <div class="llm-response-container pulse-animation">
            <h2>🤖 AI-Generated Response</h2>
            <div class="llm-content">
            {results['llm_response']}
            </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Add expandable section for prompt debugging
            if results.get('prompt_used'):
                with st.expander("🔍 View Prompt Used (Debug)"):
                    st.text(results['prompt_used'])
            
            st.markdown("---")
        
        # Main answer section
        st.markdown(f"""
        <div class="source-container">
        <h3>📄 Primary Source: <br><br>  Section {results['id']}: {results['title']}</h3>
        <p><strong>Page Number:</strong> {results.get('page_number', 'N/A')} | <strong>Query Similarity:</strong> {results['similarity_score']:.2f} | <strong>Keyword Matches:</strong> {results.get('keyword_match_count', 0)}</p>
        """, unsafe_allow_html=True)
        
        # Get keyword matches for main node
        main_matching_keywords = results.get('matching_keywords', [])
        if main_matching_keywords:
            st.markdown(f"**Matching Keywords:** {', '.join(main_matching_keywords)}")
        else:
            st.markdown("**Matching Keywords:** _No matching keywords found._")

        st.markdown("**Content of the primary Section :**")
        if results['answer']:
            st.markdown(results['answer'])
        else:
            st.markdown("_This section has no body of text._")
        
        st.markdown('</div>', unsafe_allow_html=True)
            
        # Related context section
        st.markdown('<div class="connected-nodes-container">', unsafe_allow_html=True)
        st.markdown("### 📚 Connected Nodes")

        for item in results['context']:
            try:
                # Use the query engine's built-in helper to fetch relationship details
                relationships = query_engine.get_relationship_details(results['id'], item['id'])
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
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Visualization section
        st.markdown(f"""
        <div class="graph-viz-container">
        <h3>🔍 Graph Visualization</h3>
        <p>Run this query in Neo4j Browser to visualize the graph:</p>
        <pre><code class="language-cypher">{results['visualization_query']}</code></pre>
        </div>
        """, unsafe_allow_html=True)
        
        # Close answer wrapper
        st.markdown('</div>', unsafe_allow_html=True)

def show_query_tab():
    """Display the query tab content."""
    # Sidebar with controls
    with st.sidebar:
        sidebar_container = st.container()
        with sidebar_container:
            st.markdown("### ⚙️ Query Settings")
            
            # New slider – initial retrieval size
            initial_retrieval_num = st.slider(
                "Initial number of results retrieved",
                min_value=1,
                max_value=20,
                value=1,
                help="Number of top candidate sections to show before you pick the main answer"
            )

            filter_type = st.radio(
                "Sort by / Select candidate section based on",
                ["Keyword Matches", "Similarity Score"],
                help="Choose how to sort the results"
            )

            # Main node thresholds
            st.markdown("**Initial Thresholds For Candidate Section:**")
            main_min_keyword_matches = st.slider(
                "Minimum Keyword Matches",
                min_value=1,
                max_value=10,
                value=2,
                help="Minimum number of matching keywords required for the main node"
            )
            
            main_similarity_threshold = st.slider(
                "MinimumSimilarity Threshold",
                min_value=0.00,
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
                min_value=0.00,
                max_value=1.0,
                value=0.7,
                step=0.01,
                help="Minimum similarity score required for connected nodes"
            )
            


            
            max_results = st.slider(
                "Maximum Connected Nodes",
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
                st.session_state.candidate_list = None
                st.session_state.awaiting_candidate_selection = False
                st.session_state.selected_candidate_id = None
                st.rerun()
    
    # Main query interface
    query_container = st.container()
    with query_container:
        st.markdown("""
        <div class="glass-container slide-in">
        <h2>🔍 Query Your Knowledge Graph</h2>
        <p>Ask questions about your ingested documents in natural language.
        The system will find relevant content and show how different pieces of information are connected.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ------------------------------------------------------------------
    # Data availability check – prevent queries when graph is empty
    # ------------------------------------------------------------------
    # We do this check once the header is rendered but before showing the
    # rest of the query UI.  If no nodes are present, we notify the user
    # and exit the function early.

    node_count_q, _ = 0, 0
    try:
        if 'query_engine' in st.session_state and st.session_state.query_engine:
            node_count_q, _ = get_graph_stats(st.session_state.query_engine.driver)
    except Exception as neo_err:
        # Log silently; treat as no data to avoid further errors
        print(f"Neo4j stats error (query tab): {neo_err}")

    if node_count_q == 0:
        no_data_container = st.container()
        with no_data_container:
            st.markdown('<div class="glass-container slide-in">', unsafe_allow_html=True)
            st.markdown("⚠️ **No data found in the graph.**\n\nPlease switch to the **Ingestion** tab and upload a document before querying.")
            st.markdown('</div>', unsafe_allow_html=True)
        # Early return prevents the rest of the query UI (inputs, etc.) from rendering
        return
    
    # Initialize sample query in session state if not exists
    if 'selected_sample_query' not in st.session_state:
        st.session_state.selected_sample_query = ""

    # Utility callback to copy a sample query into the text input in one run
    def _set_sample_query(text: str):
        """Populate the query text input with the chosen sample query."""
        st.session_state["query_input"] = text
        st.session_state.selected_sample_query = text

    # Query input - full width
    input_container = st.container()
    with input_container:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        query = st.text_input(
            "Enter your query:",
            placeholder="e.g., What is the imc Learning Suite?",
            key="query_input"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Sample queries in expandable section
    sample_queries_container = st.container()
    with sample_queries_container:
        with st.expander("💡 Sample Queries"):
            # Sample query 1
            st.button(
                "How can learning content be offered, and which international standards (such as SCORM, xAPI, LTI, and QTI) are supported?",
                use_container_width=True,
                on_click=_set_sample_query,
                args=(
                    "How can learning content be offered, and which international standards (such as SCORM, xAPI, LTI, and QTI) are supported?",
                ),
            )
            
            # Sample query 2
            st.button(
                "What is the imc Learning Suite?",
                use_container_width=True,
                on_click=_set_sample_query,
                args=("What is the imc Learning Suite?",),
            )

            st.button(
                "How does the system support compliance-related training and certification, and what features ensure traceability and control throughout the process?",
                use_container_width=True,
                on_click=_set_sample_query,
                args=("How does the system support compliance-related training and certification, and what features ensure traceability and control throughout the process?",),
            )
            st.button(
                "Can I customize certificates in the imc learning suite?",
                use_container_width=True,
                on_click=_set_sample_query,
                args=("Can I customize certificates in the imc learning suite?",),
            )
            st.button(
                "What languages are supported in the system?",
                use_container_width=True,
                on_click=_set_sample_query,
                args=("What languages are supported in the system?",),
            )
            
        st.markdown("")
    
    # Search button
    if st.button("🔍 Search", type="primary"):
        if query:
            # Reset candidate selection state for a fresh search
            st.session_state.candidate_list = None
            st.session_state.awaiting_candidate_selection = False
            st.session_state.selected_candidate_id = None
            st.session_state.last_results = None
            st.session_state.last_query = query

            if initial_retrieval_num > 1:
                with st.spinner("Retrieving candidate sections…"):
                    try:
                        raw_candidates = st.session_state.query_engine.get_top_candidates(
                            query,
                            limit=initial_retrieval_num,
                            filter_type=filter_type,
                            excluded_keywords=list(st.session_state.excluded_keywords)
                        )

                        # Apply main-node thresholds to candidate display list
                        if filter_type == "Keyword Matches":
                            filtered_candidates = [c for c in raw_candidates if c.get('keyword_match_count', 0) >= main_min_keyword_matches]
                        else:  # Similarity Score
                            filtered_candidates = [c for c in raw_candidates if c.get('similarity', 0) >= main_similarity_threshold]

                        if not filtered_candidates:
                            st.error("No candidates meet the specified main-node thresholds. Please relax the thresholds or reduce the initial retrieval number.")
                            return

                        st.session_state.candidate_list = filtered_candidates
                        st.session_state.awaiting_candidate_selection = True
                    except Exception as e:
                        st.error(f"Error retrieving candidates: {e}")
            else:
                # Directly compute answer (single candidate flow)
                with st.spinner("Finding the best matching sections..."):
                    try:
                        # Temporarily disable LLM if unchecked
                        if not enable_llm:
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
                            excluded_keywords=list(st.session_state.excluded_keywords),
                            candidate_limit=initial_retrieval_num
                        )

                        # Optional LLM regeneration with custom settings
                        if enable_llm and st.session_state.query_engine.openai_client and results.get('status') == 'success':
                            try:
                                main_node = {
                                    'id': results['id'],
                                    'title': results['title'],
                                    'text': results['answer'],
                                    'page_number': results.get('page_number'),
                                    'level': results.get('level')
                                }
                                prompt = st.session_state.query_engine._construct_prompt(query, main_node, results['context'])
                                llm_response = st.session_state.query_engine._call_openai_api(prompt, model=llm_model, max_tokens=max_tokens)
                                results['llm_response'] = llm_response
                                results['prompt_used'] = prompt
                            except Exception as llm_error:
                                results['llm_response'] = f"Error generating custom LLM response: {str(llm_error)}"

                        if not enable_llm:
                            st.session_state.query_engine.openai_client = original_client

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

    # ------------------------------------------------------------------
    # Candidate selection step (when initial_retrieval_num > 1)
    # ------------------------------------------------------------------
    if st.session_state.awaiting_candidate_selection and st.session_state.candidate_list and st.session_state.selected_candidate_id is None:
        st.markdown("---")
        st.markdown("## 🔍 Candidate Sections – choose the main answer")
        st.markdown("If you are not satisfied with the candidate sections, you can change the thresholds in the settings to get more sections.")

        for cand in st.session_state.candidate_list:
            # with st.expander(f"📄 {cand['title']} (ID: {cand['id']}, Page: {cand.get('page_number','N/A')})"):
            with st.expander(f"📄 Section {cand['id']} - {cand['title']}, Page: {cand.get('page_number','N/A')}"):
                st.markdown(f"**Keyword Matches:** {cand.get('keyword_match_count', 0)} , Similarity Score: {cand.get('similarity', 0)}")
                preview_text = cand.get('text', '')
                if len(preview_text) > 600:
                    preview_text = preview_text[:600] + " …"
                st.markdown(preview_text or "_No text available._")

            # Place selection button outside the expander (no nesting)
            if st.button(f"Select section {cand['id']}", key=f"select_{cand['id']}"):
                st.session_state.selected_candidate_id = cand['id']
                st.session_state.awaiting_candidate_selection = False
                st.rerun()

    # ------------------------------------------------------------------
    # If a candidate has been chosen but no results calculated yet, run final retrieval
    # ------------------------------------------------------------------
    if st.session_state.selected_candidate_id and st.session_state.last_results is None:
        with st.spinner("Generating answer from selected section…"):
            try:
                final_results = st.session_state.query_engine.get_connected_nodes(
                    query,
                    main_min_keyword_matches=main_min_keyword_matches,
                    main_similarity_threshold=main_similarity_threshold,
                    connected_min_keyword_matches=connected_min_keyword_matches,
                    connected_similarity_threshold=connected_similarity_threshold,
                    filter_type=filter_type,
                    max_results=max_results,
                    excluded_keywords=list(st.session_state.excluded_keywords),
                    candidate_limit=initial_retrieval_num,
                    preselected_main_node_id=st.session_state.selected_candidate_id
                )

                # Optionally regenerate LLM response
                if enable_llm and st.session_state.query_engine.openai_client and final_results.get('status') == 'success':
                    try:
                        mn = {
                            'id': final_results['id'],
                            'title': final_results['title'],
                            'text': final_results['answer'],
                            'page_number': final_results.get('page_number'),
                            'level': final_results.get('level')
                        }
                        prmpt = st.session_state.query_engine._construct_prompt(query, mn, final_results['context'])
                        resp = st.session_state.query_engine._call_openai_api(prmpt, model=llm_model if enable_llm else "gpt-4o-mini", max_tokens=max_tokens if enable_llm else 1500)
                        final_results['llm_response'] = resp
                        final_results['prompt_used'] = prmpt
                    except Exception as err:
                        final_results['llm_response'] = f"Error generating custom LLM response: {err}"

                # Immediately display the answer
                if final_results.get('status') == 'error':
                    st.error(final_results['answer'])
                else:
                    display_results(final_results, st.session_state.query_engine)

                st.session_state.last_results = final_results
            except Exception as fin_err:
                st.error(f"Error generating final answer: {fin_err}")

    # ------------------------------------------------------------------
    # Offer re-selection of a different candidate after showing results
    # ------------------------------------------------------------------
    if st.session_state.candidate_list and st.session_state.selected_candidate_id and len(st.session_state.candidate_list) > 1:
        st.markdown("---")
        st.markdown("### 🔁 Did you want to select another candidate?")

        for cand in st.session_state.candidate_list:
            if cand['id'] == st.session_state.selected_candidate_id:
                continue  # Skip currently selected one

            # Preview inside expander (no nested buttons)
            with st.expander(f"📄 Section {cand['id']} - {cand['title']}, Page: {cand.get('page_number','N/A')}"):
                txt = cand.get('text', '')
                if len(txt) > 600:
                    txt = txt[:600] + " …"
                st.markdown(txt or "_No text available._")

            if st.button(f"Switch to section {cand['id']}", key=f"reselect_{cand['id']}"):
                st.session_state.selected_candidate_id = cand['id']
                st.session_state.last_results = None
                st.rerun()

def show_ingestion_sidebar():
    with st.sidebar:
        # ---- your existing ingestion sidebar code ----
        ...

def show_query_sidebar():
    with st.sidebar:
        # ---- your existing query sidebar code ----
        ...

def main():
    st.set_page_config(
        page_title="GraphRAG - Document Intelligence",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    load_css()
    initialize_session_state()

    # ------------------------------------------------------------------
    # 1. Navigation widget – this is what sets the "active tab".
    # ------------------------------------------------------------------
    page = st.radio(
        "Navigation",
        ("📁 Ingestion", "🔍 Query"),
        horizontal=True,
        label_visibility="collapsed",
        key="navigation_radio"
    )

    # ------------------------------------------------------------------
    # 2. Build sidebar *before* the main area, based on 'page'.
    # ------------------------------------------------------------------
    if page == "📁 Ingestion":
        show_ingestion_sidebar()
    else:
        show_query_sidebar()

    # ------------------------------------------------------------------
    # 3. Main area.
    # ------------------------------------------------------------------
    if page == "📁 Ingestion":
        show_ingestion_tab()
    else:
        show_query_tab()

if __name__ == "__main__":
    main()
