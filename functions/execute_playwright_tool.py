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


class ExecutePlaywrightInput(BaseModel):
    """
    Input model for the execute_playwright_tool function.
    """

    playwright_command: PlaywrightCommand
    locators: List[str] = []
    attribute: str = None
    value: str = None
    url: str = None
    key: str = None


@cl.step(type="tool")
async def execute_playwright_tool(input_: ExecutePlaywrightInput):
    """
    Executes a playwright command against a page and returns the result.
    If multiple locators are provided, we will assume that the order of the locators details a parent-child relationship.
    We will locate each subsequent element within the previously located element.
    """
    page = cl.user_session.get("page")  # type: Page
    # if locators doesn't exist, just make it empty
    if not input_.locators:
        input_.locators = []
    element = None
    for locator in input_.locators:
        if element is None:
            element = page.locator(locator)
        else:
            element = element.locator(locator)
    if input_.playwright_command == PlaywrightCommand.GO_TO:
        result = await page.goto(input_.url)
        return result.url
    elif input_.playwright_command == PlaywrightCommand.GET_TEXT:
        return await element.text()
    elif input_.playwright_command == PlaywrightCommand.GET_HTML:
        return await element.inner_html()
    elif input_.playwright_command == PlaywrightCommand.GET_ATTRIBUTE:
        return await element.get_attribute(input_.attribute)
    elif input_.playwright_command == PlaywrightCommand.CLICK:
        try:
            await element.first.click()
            return "Clicked element."
        except Exception as e:
            return str(e)
    elif input_.playwright_command == PlaywrightCommand.FILL:
        try:
            await page.fill(locator, input_.value, strict=False)
            return "Filled element."
        except Exception as e:
            return str(e)
    elif input_.playwright_command == PlaywrightCommand.SELECT:
        try:
            await element.first.select_option(input_.value)
            return "Selected option."
        except Exception as e:
            return str(e)
    elif input_.playwright_command == PlaywrightCommand.HOVER:
        try:
            await element.first.hover()
            return "Hovered over element."
        except Exception as e:
            return str(e)
    elif input_.playwright_command == PlaywrightCommand.SCROLL:
        try:
            await element.first.scroll_into_view_if_needed()
            return "Scrolled to element."
        except Exception as e:
            return str(e)
    elif input_.playwright_command == PlaywrightCommand.SEND_KEYS:
        try:
            await element.send_keys(input_.value)
            return "Sent keys to element."
        except Exception as e:
            return str(e)
    elif input_.playwright_command == PlaywrightCommand.PRESS:
        try:
            await page.press(element, input_.key)
            return "Pressed key."
        except Exception as e:
            return str(e)
    elif input_.playwright_command == PlaywrightCommand.WAIT_FOR:
        try:
            await page.wait_for_timeout(input_.value)
            return "Waited for element."
        except Exception as e:
            return str(e)
    elif input_.playwright_command == PlaywrightCommand.WAIT_FOR_SELECTOR:
        try:
            await page.wait_for_selector(locator)
            return "Waited for selector."
        except Exception as e:
            return str(e)
    elif input_.playwright_command == PlaywrightCommand.WAIT_FOR_TIMEOUT:
        try:
            await page.wait_for_timeout(input_.value)
            return "Waited for timeout."
        except Exception as e:
            return str(e)
    elif input_.playwright_command == PlaywrightCommand.WAIT_FOR_NAVIGATION:
        try:
            await page.wait_for_navigation()
            return "Waited for navigation."
        except Exception as e:
            return str(e)
    elif input_.playwright_command == PlaywrightCommand.WAIT_FOR_REQUEST:
        try:
            await page.wait_for_request(input_.value)
            return "Waited for request."
        except Exception as e:
            return str(e)
    elif input_.playwright_command == PlaywrightCommand.WAIT_FOR_RESPONSE:
        try:
            await page.wait_for_response(input_.value)
            return "Waited for response."
        except Exception as e:
            return str(e)
    elif input_.playwright_command == PlaywrightCommand.WAIT_FOR_LOAD_STATE:
        try:
            await page.wait_for_load_state()
            return "Waited for load state."
        except Exception as e:
            return str(e)
        


if __name__ == "__main__":
    import json
    # Print the format that OpenAI expects
    schema = {
            "name": "execute_playwright_tool",
            "description": "Executes a playwright command against a page and returns the result.",
            "parameters": ExecutePlaywrightInput.model_json_schema(),
        }
    #print with double quotes
    with open("schema.json", "w") as f:
        f.write(json.dumps(schema, indent=4))