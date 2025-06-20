import json
from pathlib import Path
from .models import Document

def save_to_json(doc: Document, output_path: str) -> None:
    """Save document to JSON file.
    
    Args:
        doc: Document object to save
        output_path: Path to output JSON file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'title': doc.title,
            'sections': [vars(section) for section in doc.sections]
        }, f, indent=2, ensure_ascii=False)

def print_document_summary(doc: Document) -> None:
    """Print a summary of the parsed document.
    
    Args:
        doc: Document object to summarize
    """
    print(f"\n📄 Document: '{doc.title}'")
    print(f"📊 Total sections: {len(doc.sections)}")
    
    # Group sections by level
    level_counts = {}
    for section in doc.sections:
        level_counts[section.level] = level_counts.get(section.level, 0) + 1
    
    print("📋 Section breakdown:")
    for level in sorted(level_counts.keys()):
        print(f"   Level {level}: {level_counts[level]} sections")

def print_section_preview(doc: Document, max_sections: int = 5) -> None:
    """Print a preview of the first few sections.
    
    Args:
        doc: Document object
        max_sections: Maximum number of sections to preview
    """
    print(f"\n🔍 First {min(max_sections, len(doc.sections))} sections:")
    
    for section in doc.sections[:max_sections]:
        indent = "  " * (section.level - 1)
        text_preview = section.text[:100] + "..." if len(section.text) > 100 else section.text
        
        print(f"{indent}📑 {section.id} - '{section.title}'")
        print(f"{indent}   Page: {section.page_number} | TOC: {section.is_toc_entry}")
        if text_preview.strip():
            print(f"{indent}   Text: {text_preview}")
        print()

def validate_document_structure(doc: Document) -> bool:
    """Validate the document structure for consistency.
    
    Args:
        doc: Document object to validate
        
    Returns:
        True if structure is valid
    """
    issues = []
    
    # Check for empty titles
    empty_titles = [s.id for s in doc.sections if not s.title.strip()]
    if empty_titles:
        issues.append(f"Sections with empty titles: {empty_titles}")
    
    # Check for sections without text
    empty_sections = [s.id for s in doc.sections if not s.text.strip()]
    if empty_sections:
        issues.append(f"Sections with empty text: {empty_sections}")
    
    # Check parent-child relationships
    section_ids = {s.id for s in doc.sections}
    for section in doc.sections:
        if section.parent_id and section.parent_id not in section_ids:
            issues.append(f"Section {section.id} has invalid parent_id: {section.parent_id}")
    
    if issues:
        print("⚠️  Structure validation issues found:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("✅ Document structure validation passed")
        return True 