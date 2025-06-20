#!/usr/bin/env python3
"""
PDF Parser - Simple Interface

A clean command-line interface for extracting structured content from PDF documents.

Usage:
    python parse_pdf.py

Configuration:
    Update the PDF_PATH and OUTPUT_PATH variables below
"""

from pdf_reader import extract_pdf_content

# Configuration
PDF_PATH = "data/imc_LS_Functions-in-Detail_14.24_EN.pdf"
OUTPUT_PATH = "data/structured_content.json"

def main():
    """Main entry point for the PDF parser."""
    print("🚀 Starting PDF parsing...")
    print(f"📁 Input file: {PDF_PATH}")
    print(f"💾 Output file: {OUTPUT_PATH}")
    print("-" * 50)
    
    # Use the simple interface
    success = extract_pdf_content(PDF_PATH, OUTPUT_PATH)
    
    if success:
        print(f"✅ Successfully parsed PDF and saved to {OUTPUT_PATH}")
    else:
        print("❌ Failed to parse PDF")

if __name__ == "__main__":
    main()
