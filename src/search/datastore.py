from collections import OrderedDict
from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1 as discoveryengine
from typing import List, Optional


def search_datastore(
    project_id: str,
    location: str,
    engine_id: str,
    search_query: str,
    query_filter: Optional[str] = None,
    model_version: str = "preview",
    summary_result_count: int = 5,
    include_citations: bool = True,
    ignore_adversarial_query: bool = True,
    ignore_non_summary_seeking_query: bool = True,
    custom_prompt: Optional[str] = None
) -> discoveryengine.SearchResponse:
    #  For more information, refer to:
    # https://cloud.google.com/generative-ai-app-builder/docs/locations#specify_a_multi-region_for_your_data_store
    client_options = (
        ClientOptions(api_endpoint=f"{location}-discoveryengine.googleapis.com")
        if location != "global"
        else None
    )

    # Create a client
    client = discoveryengine.SearchServiceClient(client_options=client_options)

    # The full resource name of the search app serving config
    serving_config = f"projects/{project_id}/locations/{location}/collections/default_collection/engines/{engine_id}/servingConfigs/default_config"

    # Optional: Configuration options for search
    # Refer to the `ContentSearchSpec` reference for all supported fields:
    # https://cloud.google.com/python/docs/reference/discoveryengine/latest/google.cloud.discoveryengine_v1.types.SearchRequest.ContentSearchSpec
    content_search_spec = discoveryengine.SearchRequest.ContentSearchSpec(
        # For information about snippets, refer to:
        # https://cloud.google.com/generative-ai-app-builder/docs/snippets
        snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
            return_snippet=True
        ),
        # For information about search summaries, refer to:
        # https://cloud.google.com/generative-ai-app-builder/docs/get-search-summaries
        summary_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec(
            summary_result_count=summary_result_count,
            include_citations=include_citations,
            ignore_adversarial_query=ignore_adversarial_query,
            ignore_non_summary_seeking_query=ignore_non_summary_seeking_query,
            model_prompt_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec.ModelPromptSpec(
                preamble=custom_prompt
            ),
            model_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec.ModelSpec(
                version=model_version,
            ),
        ),
    )

    # Refer to the `SearchRequest` reference for all supported fields:
    # https://cloud.google.com/python/docs/reference/discoveryengine/latest/google.cloud.discoveryengine_v1.types.SearchRequest
    request = discoveryengine.SearchRequest(
        serving_config=serving_config,
        query=search_query,
        filter=query_filter,
        page_size=10,
        content_search_spec=content_search_spec,
        query_expansion_spec=discoveryengine.SearchRequest.QueryExpansionSpec(
            condition=discoveryengine.SearchRequest.QueryExpansionSpec.Condition.AUTO,
        ),
        spell_correction_spec=discoveryengine.SearchRequest.SpellCorrectionSpec(
            mode=discoveryengine.SearchRequest.SpellCorrectionSpec.Mode.AUTO
        ),
    )

    response = client.search(request)

    return response


def extract_relevant_documents_and_pages(response):
    """Extract relevant documents and pages."""
    relevant_documents_and_pages_dict = OrderedDict()
    for result in response.results:
        struct_data = dict(result.document.derived_struct_data)
        link = struct_data["link"]
        html_title = struct_data.get("htmlTitle")
        title = struct_data.get("title")
        output_dict = {}
        output_dict["title"] = title
        output_dict["html_title"] = html_title
        output_dict["link"] = link
        relevant_documents_and_pages_dict[title or html_title or link] = output_dict

    return relevant_documents_and_pages_dict
