from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Section:
    """Represents a document section with hierarchical structure."""
    id: str
    title: str
    level: int
    parent_id: Optional[str]
    text: str
    page_number: int
    is_toc_entry: bool = False

@dataclass
class Document:
    """Represents a complete document with title and sections."""
    title: str
    sections: List[Section] 