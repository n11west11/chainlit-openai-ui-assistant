"""
Given a page, command, 
"""

from unittest import result
import chainlit as cl
from enum import Enum
from typing import List
from httpx import get
from openai import BaseModel
from playwright.async_api import Page
from langchain.tools import tool
import re
from config import settings


class PlaywrightCommand(str, Enum):
    """
    Enum for Playwright commands.
    """

    GO_TO = "go_to"
    GET_TEXT = "get_text"
    GET_HTML = "get_html"
    GET_ATTRIBUTE = "get_attribute"
    CLICK = "click"
    FILL = "fill"
    SELECT = "select"
    HOVER = "hover"
    SCROLL = "scroll"
    WAIT_FOR = "wait_for"
    WAIT_FOR_SELECTOR = "wait_for_selector"
    WAIT_FOR_TIMEOUT = "wait_for_timeout"
    WAIT_FOR_NAVIGATION = "wait_for_navigation"
    WAIT_FOR_REQUEST = "wait_for_request"
    WAIT_FOR_RESPONSE = "wait_for_response"
    WAIT_FOR_LOAD_STATE = "wait_for_load_state"
    WAIT_FOR_SELECTOR_TIMEOUT = "wait_for_selector_timeout"
    SEND_KEYS = "send_keys"
    PRESS = "press"


def parse_input(value: str) -> str:
    """
    Parses the input string for $ symbols and replaces them with values from settings.
    """
    pattern = re.compile(r'\$([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*)')
    matches = pattern.findall(value)
    for match in matches:
        replacement = getattr(settings, match)
        value = value.replace(f'${match}', replacement)
    return value


async def execute_playwright_tool(
    playwright_command: PlaywrightCommand,
    locators: List[str] = None,
    attribute: str = None,
    value: str = None,
    url: str = None,
    key: str = None):

    """
    Executes a playwright command against a page and returns the result.
    If multiple locators are provided, we will assume that the order of the locators details a parent-child relationship.
    We will locate each subsequent element within the previously located element.
    """
    page = cl.user_session.get("page")  # type: Page
    locators = locators or []  # Initialize empty list if None
    element = None
    for locator in locators:
        if element is None:
            element = page.locator(locator)
        else:
            element = element.locator(locator)
    if value:
        value = parse_input(value)
    if url:
        url = parse_input(url)
    if key:
        key = parse_input(key)
    if playwright_command == PlaywrightCommand.GO_TO:
        result = await page.goto(url)
        return result.url
    elif playwright_command == PlaywrightCommand.GET_TEXT:
        return await element.text()
    elif playwright_command == PlaywrightCommand.GET_HTML:
        return await element.inner_html()
    elif playwright_command == PlaywrightCommand.GET_ATTRIBUTE:
        return await element.get_attribute(attribute)
    elif playwright_command == PlaywrightCommand.CLICK:
        try:
            await element.first.click()
            return "Clicked element."
        except Exception as e:
            return str(e)
    elif playwright_command == PlaywrightCommand.FILL:
        try:
            await page.fill(locator, value, strict=False)
            return "Filled element."
        except Exception as e:
            return str(e)
    elif playwright_command == PlaywrightCommand.SELECT:
        try:
            await element.first.select_option(value)
            return "Selected option."
        except Exception as e:
            return str(e)
    elif playwright_command == PlaywrightCommand.HOVER:
        try:
            await element.first.hover()
            return "Hovered over element."
        except Exception as e:
            return str(e)
    elif playwright_command == PlaywrightCommand.SCROLL:
        try:
            await element.first.scroll_into_view_if_needed()
            return "Scrolled to element."
        except Exception as e:
            return str(e)
    elif playwright_command == PlaywrightCommand.SEND_KEYS:
        try:
            await element.send_keys(value)
            return "Sent keys to element."
        except Exception as e:
            return str(e)
    elif playwright_command == PlaywrightCommand.PRESS:
        try:
            await page.press(element, key)
            return "Pressed key."
        except Exception as e:
            return str(e)
    elif playwright_command == PlaywrightCommand.WAIT_FOR:
        try:
            await page.wait_for_timeout(value)
            return "Waited for element."
        except Exception as e:
            return str(e)
    elif playwright_command == PlaywrightCommand.WAIT_FOR_SELECTOR:
        try:
            await page.wait_for_selector(locator)
            return "Waited for selector."
        except Exception as e:
            return str(e)
    elif playwright_command == PlaywrightCommand.WAIT_FOR_TIMEOUT:
        try:
            await page.wait_for_timeout(value)
            return "Waited for timeout."
        except Exception as e:
            return str(e)
    elif playwright_command == PlaywrightCommand.WAIT_FOR_NAVIGATION:
        try:
            await page.wait_for_navigation()
            return "Waited for navigation."
        except Exception as e:
            return str(e)
    elif playwright_command == PlaywrightCommand.WAIT_FOR_REQUEST:
        try:
            await page.wait_for_request(value)
            return "Waited for request."
        except Exception as e:
            return str(e)
    elif playwright_command == PlaywrightCommand.WAIT_FOR_RESPONSE:
        try:
            await page.wait_for_response(value)
            return "Waited for response."
        except Exception as e:
            return str(e)
    elif playwright_command == PlaywrightCommand.WAIT_FOR_LOAD_STATE:
        try:
            await page.wait_for_load_state()
            return "Waited for load state."
        except Exception as e:
            return str(e)


def open_ai_representation() -> dict:
    """
    Returns a dictionary representation of the function that can be used in an OpenAI assistant.
    """
    return {
        "name": "execute_playwright_tool",
        "description": "Executes a playwright command against a page and returns the result. "
                       "You can use environment variables in all parameters by prefixing them with a $ symbol."
                       "(i.e. $USER.EMAIL).",
        "strict": True,
        "parameters": {
            "type": "object",
            "required": [
                "playwright_command",
                "locators",
                "attribute",
                "value",
                "url",
                "key",
            ],
            "properties": {
                "playwright_command": {
                    "type": "string",
                    "description": "Command to execute using Playwright.",
                    "enum": [
                        "go_to",
                        "get_text",
                        "get_html",
                        "get_attribute",
                        "click",
                        "fill",
                        "select",
                        "hover",
                        "scroll",
                        "wait_for",
                        "wait_for_selector",
                        "wait_for_timeout",
                        "wait_for_navigation",
                        "wait_for_request",
                        "wait_for_response",
                        "wait_for_load_state",
                        "wait_for_selector_timeout",
                        "send_keys",
                        "press",
                    ],
                },
                "locators": {
                    "type": "array",
                    "description": "List of locators to identify elements on the page.",
                    "items": {
                        "type": "string",
                        "description": "A locator string for an element on the page.",
                    },
                },
                "attribute": {
                    "type": "string",
                    "description": "The attribute to retrieve from an element, applicable for GET_ATTRIBUTE command.",
                },
                "value": {
                    "type": "string",
                    "description": "The value to use with commands like FILL and SELECT.",
                },
                "url": {
                    "type": "string",
                    "description": "The URL to navigate to, used with the GO_TO command.",
                },
                "key": {
                    "type": "string",
                    "description": "The key to press on the element, used with the PRESS command.",
                },
            },
            "additionalProperties": False,
        },
    }
