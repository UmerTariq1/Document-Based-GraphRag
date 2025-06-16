import fitz
from typing import Set
from collections import Counter
from .text_utils import is_footnote

def identify_headers_footers(doc: fitz.Document) -> Set[str]:
    """Identify repeated text that appears on multiple pages (headers/footers).
    
    Args:
        doc: PyMuPDF document object
        
    Returns:
        Set of text strings that appear to be headers or footers
    """
    text_frequency = Counter()
    
    # Collect text from top and bottom of each page
    for page_num in range(min(10, len(doc))):  # Sample first 10 pages
        page = doc[page_num]
        text = page.get_text()
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        if lines:
            # Check first and last few lines of each page
            for line in lines[:3] + lines[-3:]:
                if len(line) > 10:  # Only consider substantial text
                    text_frequency[line] += 1
    
    # Find text that appears on multiple pages (likely headers/footers)
    headers_footers = set()
    for text, count in text_frequency.items():
        if count >= 3:  # Appears on at least 3 pages
            headers_footers.add(text)
    
    return headers_footers

def should_skip_line(line: str, headers_footers: Set[str]) -> bool:
    """Check if a line should be skipped (header, footer, footnote, etc.).
    
    Args:
        line: Text line to evaluate
        headers_footers: Set of identified headers/footers
        
    Returns:
        True if line should be skipped
    """
    line = line.strip()
    
    # Skip empty lines
    if not line:
        return True
    
    # Skip headers/footers
    if line in headers_footers:
        return True
    
    # Skip footnotes
    if is_footnote(line):
        return True
    
    # Skip page numbers (standalone numbers)
    import re
    if re.match(r'^\d+$', line):
        return True
    
    # Skip very short lines that are likely artifacts
    if len(line) < 3:
        return True
    
    return False 