from typing import Tuple
from neo4j import Driver


def get_graph_stats(driver: Driver) -> Tuple[int, int]:
    """Return (node_count, relationship_count) for the connected Neo4j DB.

    Parameters
    ----------
    driver : neo4j.Driver
        An active Neo4j driver instance.

    Returns
    -------
    (int, int)
        Total number of nodes and relationships respectively.  If the
        query fails (e.g., DB down) we return (0, 0).
    """
    try:
        with driver.session() as session:
            node_count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        return int(node_count), int(rel_count)
    except Exception:
        return 0, 0


def clear_database(driver: Driver) -> None:
    """Detach-delete **all** nodes and relationships in the database.

    Meant for test/ingestion resets.  Swallows exceptions and prints a
    short marker so callers can keep their own logging style.
    """
    try:
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
    except Exception:
        # Intentionally silent; caller may not care during startup scripts
        pass


def create_basic_indexes(driver: Driver, labels: Tuple[str, ...] = ("Section", "SubSection", "SubSubSection")) -> None:
    """Create the ID/level/text/title indexes used by GraphRAG.

    The Cypher clauses use the `IF NOT EXISTS` guard so the helper is safe
    to call multiple times.
    """
    with driver.session() as session:
        for label in labels:
            session.run(f"CREATE INDEX {label.lower()}_id_index IF NOT EXISTS FOR (s:{label}) ON (s.id)")
            session.run(f"CREATE INDEX {label.lower()}_level_index IF NOT EXISTS FOR (s:{label}) ON (s.level)")
            session.run(f"CREATE TEXT INDEX {label.lower()}_text_index IF NOT EXISTS FOR (s:{label}) ON (s.text)")
            session.run(f"CREATE TEXT INDEX {label.lower()}_title_index IF NOT EXISTS FOR (s:{label}) ON (s.title)") 