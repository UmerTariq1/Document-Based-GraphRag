import re
from typing import Optional

def extract_section_number(text: str) -> Optional[str]:
    """Extract section number from text if it exists.
    
    Args:
        text: Input text line
        
    Returns:
        Section number (e.g., "1.3.1") or None if not found
    """
    text = text.strip()
    
    # Skip lines that look like bullet points with parentheses: "1)", "2)", etc.
    if re.match(r'^\d+\)', text):
        return None
    
    # Skip lines that start with bullet point markers
    if text.startswith(('–', '-', '•', '*', '◦', '○')):
        return None
    
    # Match clean section numbers followed by space: "1 Title", "1.2 Title", "1.2.3 Title"
    match = re.match(r'^(\d+(?:\.\d+)*)\s', text)
    if match:
        return match.group(1)
    
    # Also match if the entire line is just a section number (for multi-line headers): "1", "1.2"
    match = re.match(r'^(\d+(?:\.\d+)*)$', text)
    if match:
        return match.group(1)
    
    return None

def get_section_level(section_id: str) -> int:
    """Determine the hierarchical level of a section based on its ID.
    
    Args:
        section_id: Section identifier (e.g., "1.3.1")
        
    Returns:
        Level depth (e.g., 3 for "1.3.1")
    """
    return len(section_id.split('.'))

def get_parent_id(section_id: str) -> Optional[str]:
    """Get the parent section ID for a given section ID.
    
    Args:
        section_id: Section identifier (e.g., "1.3.1")
        
    Returns:
        Parent section ID (e.g., "1.3" for "1.3.1") or None for top-level
    """
    parts = section_id.split('.')
    if len(parts) == 1:
        return None
    return '.'.join(parts[:-1])

def extract_section_title(line: str, section_id: str) -> str:
    """Extract the title part after the section number from a line.
    
    Args:
        line: Full line containing section number and title
        section_id: The section number part
        
    Returns:
        Cleaned title text
    """
    # Remove the section number from the beginning
    remaining = line[len(section_id):].strip()
    
    # Remove any leading separators like dots, spaces, dashes, colons
    title = re.sub(r'^[\.\s\-:]+', '', remaining).strip()
    
    return title

def clean_text(text: str) -> str:
    """Clean and normalize text content.
    
    Args:
        text: Raw text content
        
    Returns:
        Cleaned and normalized text
    """
    # Remove multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    # Remove standalone page numbers
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*\d+\s*$', '', text)
    return text.strip()

def is_footnote(line: str) -> bool:
    """Check if a line is likely a footnote.
    
    Args:
        line: Text line to check
        
    Returns:
        True if line appears to be a footnote
    """
    footnote_patterns = [
        r'^\d+\s+',  # Starting with number and space
        r'^\*\s+',   # Starting with asterisk
        r'^\[\d+\]', # [1], [2], etc.
        r'^Note:',   # Starting with "Note:"
        r'^©',       # Copyright symbols
    ]
    
    for pattern in footnote_patterns:
        if re.match(pattern, line.strip()):
            return True
    
    return False 