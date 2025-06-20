"""
PDF Reader Package

A clean interface for parsing PDF documents into structured content.

Simple Usage:
    from pdf_reader import extract_pdf_content
    
    extract_pdf_content("input.pdf", "output.json")
"""

from .pdf_parser import parse_pdf
from .output_utils import save_to_json

def extract_pdf_content(input_path: str, output_path: str) -> bool:
    """
    Simple interface to extract PDF content and save to JSON.
    
    Args:
        input_path: Path to the input PDF file
        output_path: Path to save the output JSON file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Parse the PDF
        document = parse_pdf(input_path)
        
        # Save to JSON
        save_to_json(document, output_path)
        
        return True
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return False

# For backward compatibility
__all__ = ["extract_pdf_content", "parse_pdf", "save_to_json"] 