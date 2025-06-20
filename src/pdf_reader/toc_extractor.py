import fitz
import re
from typing import Dict
from .header_detection import should_skip_line

def extract_toc_entries(doc: fitz.Document, headers_footers: set) -> Dict[str, Dict[str, any]]:
    """Extract Table of Contents entries from the first few pages of the document.
    
    Args:
        doc: PyMuPDF document object
        headers_footers: Set of identified headers/footers to skip
        
    Returns:
        Dictionary mapping section IDs to their TOC information
    """
    toc_entries = {}
    # Pattern for single-line entries like "1 The imc Learning Suite 4"
    single_line_regex = re.compile(r'^(\d+(?:\.\d+)*)\s+(.*?)\s*\.*\s*(\d+)$')
    # Pattern for just a section number
    section_num_regex = re.compile(r'^(\d+(?:\.\d+)*)$')
    # Pattern for just a page number
    page_num_regex = re.compile(r'^\d+$')

    for page_num in range(min(3, len(doc))):
        page = doc[page_num]
        text = page.get_text()
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            if line in headers_footers:
                i += 1
                continue
            
            lines_consumed = 1
            print("line", line)
            
            # Try single-line pattern first
            single_line_match = single_line_regex.match(line)
            if single_line_match:
                section_id, title, page_str = single_line_match.groups()
                try:
                    page_int = int(page_str)
                    toc_entries[section_id] = {'title': title, 'page': page_int}
                    print(f"✓ TOC (Single-line): {section_id} - '{title}' (page {page_int})")
                except ValueError:
                    pass
            # Handle split entries (number on one line, title on next, page on third)
            elif section_num_regex.match(line) and i + 1 < len(lines):
                section_id = line
                title = lines[i + 1]
                
                # Look ahead for page number
                if i + 2 < len(lines) and page_num_regex.match(lines[i + 2]):
                    try:
                        page_int = int(lines[i + 2])
                        toc_entries[section_id] = {'title': title, 'page': page_int}
                        print(f"✓ TOC (Split): {section_id} - '{title}' (page {page_int})")
                        lines_consumed = 3
                    except ValueError:
                        pass
                # If no page number found, this might be a false positive
                else:
                    print(f"⚠️ Skipping potential false positive: {section_id} - '{title}'")
            # Handle case where we have a title followed by a page number
            elif not section_num_regex.match(line) and not page_num_regex.match(line) and i + 1 < len(lines):
                title = line
                if page_num_regex.match(lines[i + 1]):
                    try:
                        page_int = int(lines[i + 1])
                        # Look back for section number
                        if i > 0 and section_num_regex.match(lines[i - 1]):
                            section_id = lines[i - 1]
                            toc_entries[section_id] = {'title': title, 'page': page_int}
                            print(f"✓ TOC (Split-back): {section_id} - '{title}' (page {page_int})")
                            lines_consumed = 2
                    except ValueError:
                        pass
            
            i += lines_consumed
    
    return toc_entries

def extract_document_title(doc: fitz.Document, headers_footers: set) -> str:
    """Extract the document title from the first page.
    
    Args:
        doc: PyMuPDF document object
        headers_footers: Set of identified headers/footers to skip
        
    Returns:
        Document title or None if not found
    """
    # Look for title in first page
    page = doc[0]
    text = page.get_text()
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if line and not should_skip_line(line, headers_footers):
            return line
    
    return None 