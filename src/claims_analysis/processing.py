import re
from typing import Any, Dict


def structure_claims_analysis(data: Dict[str, Any]) -> Dict[str, Any]:
    """Process API output."""
    # Extract the main text and grounding supports
    text = data['candidates'][0]['content']['parts'][0]['text']
    if not data['candidates'][0].get('grounding_metadata', {}).get('grounding_supports'):
        return {}
    grounding_supports = data['candidates'][0]['grounding_metadata']['grounding_supports']
    grounding_chunks = data['candidates'][0]['grounding_metadata']['grounding_chunks']

    # Sort grounding supports by start index
    grounding_supports.sort(key=lambda x: x['segment'].get('start_index', 0))

    # Process text and add citations with confidence scores
    processed_text = ""
    last_end = 0
    for support in grounding_supports:
        start = support['segment'].get('start_index', 0)
        end = support['segment']['end_index']
        citation_indices = [index + 1 for index in support['grounding_chunk_indices']]  # Add 1 to each index
        confidence_score = support['confidence_scores'][0]  # All scores are the same, so we take the first one

        processed_text += text[last_end:start]
        processed_text += text[start:end]
        processed_text += f"[{','.join(map(str, citation_indices))}][{confidence_score:.2f}]"

        last_end = end

    processed_text += text[last_end:]

    # Generate citations list
    citations = []
    for i, chunk in enumerate(grounding_chunks, start=1):  # Start enumeration from 1
        title = chunk['web']['title']
        uri = chunk['web']['uri']
        citations.append(f"{i}. [{title}]({uri})")

    output_markdown = generate_markdown(processed_text, citations)
    structured_claims_analysis = parse_claim_analysis(output_markdown)
    return structured_claims_analysis


def generate_markdown(processed_text, citations):
    markdown = processed_text + "\n\n## Citations\n\n" + "\n".join(citations)
    return markdown


def parse_claim_analysis(text: str) -> Dict[str, Any]:
    # Split the text into claim analysis and alternatives
    parts = re.split(r'\nAlternatives:', text, maxsplit=1)
    claim_analysis = parts[0].replace('Claim Analysis:', '').strip()
    alternatives_text = parts[1] if len(parts) > 1 else ''

    # Parse alternatives
    alternatives = []
    for alt in re.split(r'\n\d+\.', alternatives_text):
        if alt.strip():
            alt_parts = alt.split('Explanation:', 1)
            if len(alt_parts) == 2:
                alternatives.append({
                    "improved_claim": alt_parts[0].strip(),
                    "explanation": alt_parts[1].strip()
                })

    # Extract citations
    citations = []
    citation_pattern = r'\d+\.\s+\[(.*?)\]\((.*?)\)'
    for match in re.finditer(citation_pattern, text):
        citations.append({
            "title": match.group(1),
            "uri": match.group(2)
        })

    # Remove citations section from claim analysis
    claim_analysis = re.sub(r'## Citations.*', '', claim_analysis, flags=re.DOTALL).strip()

    # Structure the output
    structured_output = {
        "claim_analysis": claim_analysis,
        "alternatives": alternatives,
        "citations": citations
    }

    return structured_output
