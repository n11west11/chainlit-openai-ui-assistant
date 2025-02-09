import os
import json
from io import BytesIO
from pathlib import Path
from typing import List

from numpy import save
from openai import AsyncAssistantEventHandler, AsyncOpenAI, OpenAI
from typing_extensions import override
from playwright import async_api
from playwright.async_api import Page

import chainlit as cl
from chainlit.config import config
from chainlit.element import Element
from chainlit.context import local_steps

from literalai.helper import utc_now

from functions import condense_html, execute_playwright_tool, search_page_tool, save_information

async_openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
sync_openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

assistant = sync_openai_client.beta.assistants.retrieve(
    os.environ.get("OPENAI_ASSISTANT_ID")
)
ASSISTANT_INSTRUCTIONS = """
You are a helpful UI assistant, you aid in executing commands against a browesr primarily through the execute_playwright_tool input.
You can execute instructions against the following html, always review the current html before execting your next command. If the html 
looks like it is cut off, use the search_page_tool to search the page for specific information:"""

assistant.tools = [
    {
        "type": "file_search",
    },
    {
        "type": "code_interpreter",
    },
    {
        "type": "function",
        "function": condense_html.open_ai_representation(),
    },
    {
        "type": "function",
        "function": execute_playwright_tool.open_ai_representation(),
    },
    {
        "type": "function",
        "function": search_page_tool.open_ai_representation(),
    },
    {
        "type": "function",
        "function": save_information.open_ai_representation(),
    }
]
sync_openai_client.beta.assistants.update(
    assistant_id=assistant.id,
    tools=assistant.tools,
    instructions=ASSISTANT_INSTRUCTIONS,
)

config.ui.name = assistant.name


class EventHandler(AsyncAssistantEventHandler):
    """Handles events from the OpenAI Assistant API.
    This class manages the interaction between the assistant and the user.
    It handles text input, tool calls, and image file uploads.
    """

    def __init__(self, assistant_name: str) -> None:
        super().__init__()
        self.current_message: cl.Message = None
        self.current_step: cl.Step = None
        self.current_tool_call = None
        self.assistant_name = assistant_name
        previous_steps = local_steps.get() or []
        parent_step = previous_steps[-1] if previous_steps else None
        if parent_step:
            self.parent_id = parent_step.id

    @override
    async def on_text_created(self, text) -> None:
        self.current_message = await cl.Message(
            author=self.assistant_name, content=""
        ).send()

    @override
    async def on_text_delta(self, delta, snapshot):
        await self.current_message.stream_token(delta.value)

    @override
    async def on_text_done(self, text):
        await self.current_message.update()

    async def on_tool_call_created(self, tool_call):
        self.current_tool_call = tool_call.id
        self.current_step = cl.Step(name=tool_call.type, type="tool", parent_id=self.parent_id)
        self.current_step.show_input = "python"
        self.current_step.start = utc_now()
        await self.current_step.send()


    async def on_tool_call_delta(self, delta, snapshot): 
        if snapshot.id != self.current_tool_call:
            self.current_tool_call = snapshot.id
            self.current_step = cl.Step(name=delta.type, type="tool", parent_id=self.parent_id)
            self.current_step.start = utc_now()
            if snapshot.type == "code_interpreter":
                 self.current_step.show_input = "python"
            if snapshot.type == "function":
                self.current_step.name = snapshot.function.name
                self.current_step.language = "json"
            await self.current_step.send()
        
        if delta.type == "function":
            pass
        
        if delta.type == "code_interpreter":
            if delta.code_interpreter.outputs:
                for output in delta.code_interpreter.outputs:
                    if output.type == "logs":
                        self.current_step.output += output.logs
                        self.current_step.language = "markdown"
                        self.current_step.end = utc_now()
                        await self.current_step.update()
                    elif output.type == "image":
                        self.current_step.language = "json"
                        self.current_step.output = output.image.model_dump_json()
            else:
                if delta.code_interpreter.input:
                    await self.current_step.stream_token(delta.code_interpreter.input, is_input=True) 

    @override
    async def on_tool_call_done(self, tool_call):
        self.current_step.end = utc_now()
        await self.current_step.update()
        run_status = await async_openai_client.beta.threads.runs.retrieve(
            thread_id=self.current_run.thread_id, run_id=self.current_run.id
        )

        while run_status.status in ["queued", "in_progress", "requires_action"]:
            if run_status.status == "requires_action":
                self.current_step = cl.Step(
                    name="submit_tool_outputs", type="tool", parent_id=self.current_step.id
                )
                for (
                    tool_call
                ) in run_status.required_action.submit_tool_outputs.tool_calls:
                    function = tool_call.function
                    try:
                        arguments = json.loads(function.arguments)
                    except Exception as _:
                        arguments = function.arguments
                        result = str(_)
                    if function.name == "execute_playwright_tool":
                        try:
                            # await self.current_step.stream_token(function.arguments)

                            # load arguments into ExecutePlaywrightInput model
                            await screenshot(self.current_step)
                            result = (
                                await execute_playwright_tool.execute_playwright_tool(
                                    **arguments
                                )
                            )
                            await screenshot(self.current_step)
                        except Exception as e:
                            result = str(e)

                        if not result:
                            result = "success"
                    elif function.name == "search_page_tool":
                        await screenshot()
                        result = await search_page_tool.search_page_tool(**arguments)
                    elif function.name == "save_information":
                        result = await save_information.save_information(**arguments)
                    elif function.name == "condense_html":
                        await screenshot()
                        result = await condense_html.condense_html(**arguments)
                    await async_openai_client.beta.threads.runs.submit_tool_outputs(
                        thread_id=self.current_event.data.thread_id,
                        run_id=self.current_run.id,
                        tool_outputs=[{"tool_call_id": tool_call.id, "output": result}],
                    )
                    await set_playwright_context()
                    await self.current_step.update()
                    self.current_step.end = utc_now()

            run_status = await async_openai_client.beta.threads.runs.retrieve(
                thread_id=self.current_run.thread_id, run_id=self.current_run.id
            )


        # get latest messages
        latest_messages = await async_openai_client.beta.threads.messages.list(
            thread_id=self.current_run.thread_id
        )
        await cl.Message(
            author=self.assistant_name,
            content=latest_messages.data[0].content[0].text.value,
        ).send()

    async def on_image_file_done(self, image_file):
        image_id = image_file.file_id
        response = await async_openai_client.files.with_raw_response.content(image_id)
        image_element = cl.Image(
            name=image_id, content=response.content, display="inline", size="large"
        )
        if not self.current_message.elements:
            self.current_message.elements = []
        self.current_message.elements.append(image_element)
        await self.current_message.update()


async def set_playwright_context():
    """Updates the Playwright context in the user session.
    If called, clears the current context and updates with new page content.
    Args:
        request: The request object to set as the new Playwright context. If None, clears the context.
    """
    # The assistant instructions are the only dynamic aspect of the assistant api right now.
    # It has a 250000 character limit, so we need to condense the html content to fit within that limit.
    page = cl.user_session.get("page")  # type: Page
    content = await page.content()
    condensed_html = condense_html.condense_html(content)
    if len(condensed_html) > 250000:
        print("HTML content is too long. Truncating to 250,000 characters.")
        condensed_html = condensed_html[:250000]
    await async_openai_client.beta.assistants.update(
        assistant_id=assistant.id,
        instructions=ASSISTANT_INSTRUCTIONS + condensed_html,
    )
    return page.url


@cl.step(type="tool")
async def speech_to_text(audio_file):
    """Converts audio to text using OpenAI's Whisper API.
    Args:
        audio_file: The audio file to convert to text.
    Returns:
        The text transcription of the audio file.
    """
    response = await async_openai_client.audio.transcriptions.create(
        model="whisper-1", file=audio_file
    )

    return response.text


async def upload_files(files: List[Element]):
    """Uploads files to OpenAI and returns their file IDs.
    Args:
        files: List of uploaded files to process.
    Returns:
        List of file IDs."""
    file_ids = []
    for file in files:
        uploaded_file = await async_openai_client.files.create(
            file=Path(file.path), purpose="assistants"
        )
        file_ids.append(uploaded_file.id)
    return file_ids


async def process_files(files: List[Element]):
    """Processes uploaded files.
    Uploads files to OpenAI and returns their file IDs.
    Args:
        files: List of uploaded files to process.
    Returns:
        List of dictionaries containing file IDs and their associated tools."""
    # Upload files if any and get file_ids
    file_ids = []
    if len(files) > 0:
        file_ids = await upload_files(files)

    return [
        {
            "file_id": file_id,
            "tools": [{"type": "code_interpreter"}, {"type": "file_search"}],
        }
        for file_id in file_ids
    ]


@cl.on_chat_start
async def start_chat():
    """Initializes the chat session.
    Creates a thread, sets up a playwright session, and stores the thread ID in the user session.

    To connect to an existing Chrome instance on Mac, first launch Chrome with:
    /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome --remote-debugging-port=9222 --user-data-dir=/tmp/remote-profile
    """
    # Create a Thread
    thread = await async_openai_client.beta.threads.create()

    try:
        playwright = await async_api.async_playwright().start()
        # First try launching our own browser
        print("Launching new browser instance...")
        async_browser = await playwright.chromium.launch(headless=False)
            
    except Exception as e:
        print(f"Failed to initialize browser: {e}")
        raise

    if len(async_browser.contexts) == 0:
        page = await async_browser.new_page()
    else:
        page = async_browser.contexts[0]
    page.set_default_timeout(10000)
    cl.user_session.set("page", page)
    cl.user_session.set("browser", async_browser)
    cl.user_session.set("playwright", playwright)

    # Store thread ID in user session for later use
    cl.user_session.set("thread_id", thread.id)
    cl.user_session.set("assistant", assistant)
    await cl.Message(
        content=f"Hello, I'm {assistant.name}!"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    thread_id = cl.user_session.get("thread_id")

    attachments = await process_files(message.elements)

    # Add a Message to the Thread
    await async_openai_client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=message.content,
        attachments=attachments,
    )

    # Create and Stream a Run
    async with async_openai_client.beta.threads.runs.stream(
        thread_id=thread_id,
        assistant_id=assistant.id,
        event_handler=EventHandler(assistant_name=assistant.name),
    ) as stream:
        await stream.until_done()


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    """Handles audio chunks from the user.
    If the chunk is the start of a new audio stream, it initializes the session for a new audio stream.
    Writes the chunks to a buffer and transcribes the whole audio at the end."""
    if chunk.isStart:
        buffer = BytesIO()
        # This is required for whisper to recognize the file type
        buffer.name = f"input_audio.{chunk.mimeType.split('/')[1]}"
        # Initialize the session for a new audio stream
        cl.user_session.set("audio_buffer", buffer)
        cl.user_session.set("audio_mime_type", chunk.mimeType)

    # Write the chunks to a buffer and transcribe the whole audio at the end
    cl.user_session.get("audio_buffer").write(chunk.data)

async def screenshot(current_step: cl.Step):
    """Takes a screenshot of the current page and sends it as a Step element.
    If there's an error, sends the error message as a Step element instead."""
    page = cl.user_session.get("page")
    try:
        _screenshot = cl.Image(
            name="screenshot",
            content=await page.screenshot(),
            display="inline",
            size="medium",
        )
        await cl.Step(
            name="screenshot", elements=[_screenshot], parent_id=current_step.parent_id
        ).send()
    except Exception as e:
        await cl.Step(
            name="screenshot", elements=[cl.Message(author="You", content=str(e))]
        ).send()


@cl.on_audio_end
async def on_audio_end(elements: list[Element]):
    # Get the audio buffer from the session
    audio_buffer: BytesIO = cl.user_session.get("audio_buffer")
    audio_buffer.seek(0)  # Move the file pointer to the beginning
    audio_file = audio_buffer.read()
    audio_mime_type: str = cl.user_session.get("audio_mime_type")

    input_audio_el = cl.Audio(
        mime=audio_mime_type, content=audio_file, name=audio_buffer.name
    )
    await cl.Message(
        author="You",
        type="user_message",
        content="",
        elements=[input_audio_el, *elements],
    ).send()

    whisper_input = (audio_buffer.name, audio_file, audio_mime_type)
    transcription = await speech_to_text(whisper_input)

    msg = cl.Message(author="You", content=transcription, elements=elements)

    await main(message=msg)


# Add new cleanup handler
@cl.on_chat_end
async def end_chat():
    """Cleanup browser resources when chat ends"""
    browser = cl.user_session.get("browser")
    playwright = cl.user_session.get("playwright")
    
    if browser:
        await browser.close()
    if playwright:
        await playwright.stop()