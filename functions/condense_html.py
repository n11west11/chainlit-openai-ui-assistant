""" 
This module contains function that condenses the HTML content by keeping 
only essential elements and attributes.
"""

from bs4 import BeautifulSoup, Comment


def condense_html(html: str) -> str:
    """
    Condenses the HTML content by keeping only essential elements and attributes.
    This helps in reducing the size of the HTML to be processed.

    :param html: The HTML content to be condensed.
    :return: A string representing the condensed HTML content.
    """
    soup = BeautifulSoup(html, "html.parser")
    soup = soup.body if soup.body else soup

    for tag in soup(["script", "style", "head", "meta", "link"]):
        tag.decompose()

    # Remove comments
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    for comment in comments:
        comment.extract()

    # Define allowed tags and their attributes
    default_tags = ["class", "id", "data-testid", "aria-label"]
    allowed_attr_dict = {
        "a": ["title", "name"] + default_tags,
        # 'img': ['src', 'alt'] + default_tags,
        "input": ["type", "name", "value"] + default_tags,
        "select": ["type", "name", "value"] + default_tags,
        "button": ["type", "name", "value"] + default_tags,
        "option": ["type", "name", "value"] + default_tags,
        "label": ["for"] + default_tags,
        "textarea": ["type", "name", "value"] + default_tags,
        "form": ["action"] + default_tags,
        "iframe": ["src", "title", "name"] + default_tags,
        # 'video': ['src', 'title', 'name'] + default_tags,
    }

    # Remove all tags not in allowed_tags, keep only allowed attributes for allowed tags
    for tag in soup.find_all(True):
        # Keep only allowed attributes for this tag, remove others
        allowed_attrs = (
            allowed_attr_dict[tag.name] if tag.name in allowed_attr_dict else []
        )
        for attr in list(tag.attrs):
            if attr not in allowed_attrs:
                del tag[attr]
        if tag.name not in allowed_attr_dict:
            # delete every attribute but keep the txt
            tag.attrs = {}

    # Collapse whitespace in text
    for tag in soup.find_all(string=True):
        new_text = " ".join(tag.split())
        tag.replace_with(new_text)

    # unwrap tags that have no attributes
    for tag in soup.find_all(True):
        if not tag.attrs:
            tag.unwrap()

    return str(soup)


def open_ai_representation() -> dict:
    """
    Returns a dictionary representation of the function that can be used in an OpenAI assistant.
    """
    return {
        "name": "condense_html",
        "description": "Condenses the HTML content by keeping only essential elements and attributes. This helps in reducing the size of the HTML to be processed.",
        "strict": True,
        "parameters": {
            "type": "object",
            "required": ["html"],
            "properties": {
                "html": {
                    "type": "string",
                    "description": "The HTML content to be condensed.",
                }
            },
            "additionalProperties": False,
        },
    }
