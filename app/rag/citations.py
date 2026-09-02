from app.rag.models import Citation


def format_citations(
    citations: list[Citation],
) -> str:
    """Format citations for terminal or UI display."""

    if not citations:
        return "No sources available."

    lines = ["Sources:"]

    for citation in citations:
        lines.append(
            f"[{citation.source_id}] "
            f"{citation.document_name} — "
            f"Page {citation.page_number} "
            f"(similarity: {citation.score:.4f})"
        )

    return "\n".join(lines)