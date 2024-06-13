from functions import condense_html, num_tokens_from_string
from langchain_community.document_loaders import AsyncHtmlLoader


def test_condense_html():
    """
    Test the condense_html function by loading a Wikipedia page and condensing its HTML content.
    """
    URL = "https://academybugs.com/find-bugs/"
    loader = AsyncHtmlLoader([URL])
    html = loader.load()[0].page_content
    CONDENSED_HTML = condense_html.condense_html(html)
    with open("condensed_html.html", "w", encoding="utf-8") as f:
        f.write(CONDENSED_HTML)
    print(
        f"token length: {num_tokens_from_string.num_tokens_from_string(CONDENSED_HTML)}"
    )
