import fitz
from typing import List, Tuple
from .models import Section, Document
from .text_utils import (
    extract_section_number, 
    get_section_level, 
    get_parent_id, 
    extract_section_title, 
    clean_text
)
from .header_detection import identify_headers_footers, should_skip_line
from .toc_extractor import extract_toc_entries, extract_document_title

def parse_pdf_content(doc: fitz.Document, toc_entries: dict, headers_footers: set) -> List[Section]:
    """Parse the main content of the PDF document starting from page 4.
    
    Args:
        doc: PyMuPDF document object
        toc_entries: Dictionary of TOC entries
        headers_footers: Set of headers/footers to skip
        
    Returns:
        List of parsed sections
    """
    sections = []
    current_section = None
    current_text = []
    
    # Get level 1 sections from TOC to create exceptions for filtering
    level1_toc = {id: info for id, info in toc_entries.items() if get_section_level(id) == 1}
    
    # Extract actual content starting from page 4
    for page_num in range(3, len(doc)):
        page = doc[page_num]
        text = page.get_text()
        lines = text.split('\n')
        
        # Check if a level 1 section is expected on this page
        expected_level1_id = None
        for section_id, info in level1_toc.items():
            if info['page'] == page_num + 1:
                expected_level1_id = section_id
                break
        
        # Pre-filter lines, with an exception for expected level 1 section IDs
        filtered_lines = []
        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                continue
            
            # CRITICAL FIX: Use the canonical section extractor to protect level 1 headers.
            # This ensures the protection logic is identical to the parsing logic.
            is_level1_header = False
            if expected_level1_id:
                id_from_line = extract_section_number(stripped_line)
                if id_from_line == expected_level1_id:
                    is_level1_header = True

            if is_level1_header:
                filtered_lines.append(stripped_line)
                continue
            
            if not should_skip_line(stripped_line, headers_footers):
                filtered_lines.append(stripped_line)
        
        i = 0
        while i < len(filtered_lines):
            line = filtered_lines[i]
            
            # Check if this is a section header
            section_id = extract_section_number(line)
            
            if section_id:
                # If there's a current section, save its collected text before starting a new one.
                # This handles the case where a section has no body text and is followed immediately by a sub-section.
                if current_section:
                    current_section.text = clean_text(' '.join(current_text))
                    sections.append(current_section)
                
                # Extract title using lookahead logic
                title, lines_consumed = _extract_section_title_with_lookahead(
                    filtered_lines, i, line, section_id
                )
                
                # Create the new section. It becomes the current_section.
                current_section = _create_section(
                    section_id, title, toc_entries, page_num + 1
                )
                current_text = [] # Reset text for the new section.
                
                i += lines_consumed
                
            elif current_section:
                # This is a content line, add it to the current section's text.
                current_text.append(line)
                i += 1
            else:
                # This line is content that appears before the very first section.
                # We are ignoring it, as per the established logic.
                i += 1
    
    # Add the very last section to the list
    if current_section:
        current_section.text = clean_text(' '.join(current_text))
        sections.append(current_section)
    
    return sections

def _extract_section_title_with_lookahead(filtered_lines: List[str], i: int, line: str, section_id: str) -> Tuple[str, int]:
    """Extract section title using lookahead logic for multi-line headers.
    
    Args:
        filtered_lines: List of filtered text lines
        i: Current line index
        line: Current line text
        section_id: Extracted section ID
        
    Returns:
        Tuple of (title, lines_consumed) where lines_consumed is how many lines to skip
    """
    title = ""
    lines_consumed = 1  # At minimum we consume the current line
    
    # First, try to extract title from same line (fallback for inline titles)
    inline_title = extract_section_title(line, section_id)
    if inline_title:
        title = inline_title
    else:
        # Look ahead to next line(s) for title
        lookahead = 1
        while i + lookahead < len(filtered_lines):
            next_line = filtered_lines[i + lookahead].strip()
            if next_line and not extract_section_number(next_line):  # Not another section header
                title = next_line
                lines_consumed += 1  # We consumed this line too
                break
            elif extract_section_number(next_line):  # Break early if it's a new section (failsafe)
                break
            lookahead += 1
    
    return title, lines_consumed

def _create_section(section_id: str, title: str, toc_entries: dict, page_num: int) -> Section:
    """Create a Section object with all required metadata.
    
    Args:
        section_id: Section identifier
        title: Section title
        toc_entries: Dictionary of TOC entries
        page_num: Current page number
        
    Returns:
        New Section object
    """
    level = get_section_level(section_id)
    parent_id = get_parent_id(section_id)
    final_title = title
    
    # Check if this section exists in TOC and get title from there if available
    is_toc_entry = section_id in toc_entries
    if is_toc_entry and toc_entries[section_id]['title']:
        # Use TOC title if available and current title is empty or better
        toc_title = toc_entries[section_id]['title']
        if not final_title or len(final_title) < len(toc_title):
            final_title = toc_title
    
    toc_page = toc_entries.get(section_id, {}).get('page', page_num)
    
    return Section(
        id=section_id,
        title=final_title,
        level=level,
        parent_id=parent_id,
        text="",
        page_number=toc_page,
        is_toc_entry=is_toc_entry
    )

def _validate_section_coverage(toc_entries: dict, sections: List[Section]) -> None:
    """Validate that we found all the sections that exist in the TOC.
    
    Args:
        toc_entries: Dictionary of TOC entries
        sections: List of parsed sections
    """
    # Count sections by level in TOC
    toc_level1 = [id for id in toc_entries.keys() if get_section_level(id) == 1]
    toc_level2 = [id for id in toc_entries.keys() if get_section_level(id) == 2]
    
    # Count sections by level in parsed content
    parsed_level1 = [s.id for s in sections if s.level == 1]
    parsed_level2 = [s.id for s in sections if s.level == 2]
    
    print("\n" + "="*60)
    print("📊 SECTION COVERAGE VALIDATION")
    print("="*60)
    
    print(f"\n🔍 Level 1 Sections:")
    print(f"   TOC found: {len(toc_level1)} sections {sorted(toc_level1)}")
    print(f"   Parsed:    {len(parsed_level1)} sections {sorted(parsed_level1)}")
    
    missing_level1 = set(toc_level1) - set(parsed_level1)
    if missing_level1:
        print(f"   ❌ MISSING: {sorted(missing_level1)}")
        for missing in sorted(missing_level1):
            toc_info = toc_entries[missing]
            print(f"      {missing}: '{toc_info['title']}' (page {toc_info['page']})")
    else:
        print(f"   ✅ All level 1 sections found!")
    
    print(f"\n🔍 Level 2 Sections:")
    print(f"   TOC found: {len(toc_level2)} sections")
    print(f"   Parsed:    {len(parsed_level2)} sections")
    
    missing_level2 = set(toc_level2) - set(parsed_level2)
    if missing_level2:
        print(f"   ❌ MISSING: {len(missing_level2)} sections {sorted(missing_level2)}")
    else:
        print(f"   ✅ All level 2 sections found!")
    
    print("="*60)

def parse_pdf(pdf_path: str) -> Document:
    """Main function to parse a PDF document into structured content.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Document object with title and sections
    """
    doc = fitz.open(pdf_path)
    
    # Step 1: Identify headers and footers
    headers_footers = identify_headers_footers(doc)
    print(f"Identified {len(headers_footers)} potential headers/footers")
    
    # Step 2: Extract document title
    document_title = extract_document_title(doc, headers_footers)
    
    # Step 3: Extract TOC entries
    toc_entries = extract_toc_entries(doc, headers_footers)
    print(f"Found {len(toc_entries)} TOC entries")
    
    # Step 4: Parse main content
    sections = parse_pdf_content(doc, toc_entries, headers_footers)
    
    print(f"Headers/footers identified: {list(headers_footers)}")
    print("The document has", len(sections), "sections")
    
    # Step 5: Validate section coverage
    _validate_section_coverage(toc_entries, sections)
    
    return Document(title=document_title, sections=sections) 