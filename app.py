
import os
import json
from io import BytesIO
from pathlib import Path
from typing import List

from openai import AsyncAssistantEventHandler, AsyncOpenAI, OpenAI

from literalai.helper import utc_now

import chainlit as cl
from chainlit.config import config
from chainlit.element import Element

from playwright import async_api
from playwright.async_api import Page

from functions.condense_html import condense_html
from functions.execute_playwright_tool import (
    execute_playwright_tool,
    ExecutePlaywrightInput,
)
from typing_extensions import override


async_openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
sync_openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

assistant = sync_openai_client.beta.assistants.retrieve(
    os.environ.get("OPENAI_ASSISTANT_ID")
)

config.ui.name = assistant.name

ASSISTANT_INSTRUCTIONS = """
You are a helpful UI assistant, you aid in executing commands against a browesr primarily through the execute_playwright_tool input.
You can execute instructions against the following html, only use locators that exist in this html:"""

class EventHandler(AsyncAssistantEventHandler):

    def __init__(self, assistant_name: str) -> None:
        super().__init__()
        self.current_message: cl.Message = cl.context.current_step
        self.current_step: cl.Step = None
        self.current_tool_call = None
        self.assistant_name = assistant_name

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

    @cl.step
    async def on_tool_call_created(self, tool_call):
        self.current_tool_call = tool_call.id
        self.current_step = cl.context.current_step
        self.current_step.language = "python"
        self.current_step.created_at = utc_now()
        await self.current_step.send()


        
    @cl.step
    async def on_tool_call_delta(self, delta, snapshot):
        if snapshot.id != self.current_tool_call:
            self.current_tool_call = snapshot.id
            self.current_step = cl.context.current_step
            self.current_step.language = "python"
            self.current_step.start = utc_now()
            await self.current_step.send()

        if delta.type == "code_interpreter":
            if delta.code_interpreter.outputs:
                for output in delta.code_interpreter.outputs:
                    if output.type == "logs":
                        error_step = cl.context.current_step
                        error_step.is_error = True
                        error_step.output = output.logs
                        error_step.language = "markdown"
                        error_step.start = self.current_step.start
                        error_step.end = utc_now()
                        await error_step.send()
            else:
                if delta.code_interpreter.input:
                    await self.current_step.stream_token(delta.code_interpreter.input)

        if delta.type == "function":
            await self.current_step.stream_token(delta.function.arguments)

    @override
    async def on_tool_call_done(self, tool_call):
        self.current_step.end = utc_now()
        await self.current_step.update()
        run_status = await async_openai_client.beta.threads.runs.retrieve(
            thread_id=self.current_run.thread_id, run_id=self.current_run.id
        )

        while run_status.status in ["queued", "in_progress", "requires_action"]:
            if run_status.status == "requires_action":
                for tool_call in  run_status.required_action.submit_tool_outputs.tool_calls:
                    function = tool_call.function
                    if function.name == "execute_playwright_tool":
                        try:
                            # await self.current_step.stream_token(function.arguments)
                            arguments = json.loads(function.arguments)
                            # load arguments into ExecutePlaywrightInput model
                            arguments = ExecutePlaywrightInput(**arguments)
                            await screenshot()
                            result = await execute_playwright_tool(arguments)
                            await screenshot()
                        except Exception as e:
                            result = str(e)

                        if not result:
                            result = "success"

                        await async_openai_client.beta.threads.runs.submit_tool_outputs(
                            thread_id=self.current_event.data.thread_id,
                            run_id=self.current_run.id,
                            tool_outputs=[{"tool_call_id": tool_call.id, "output": result}],
                        )
                        content = await set_playwright_context(None)
                        
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


@cl.step
async def set_playwright_context(request):
    # The assistant instructions are the only dynamic aspect of the assistant api right now. 
    # It has a 250000 character limit, so we need to condense the html content to fit within that limit.
    page = cl.user_session.get("page")  # type: Page
    content = await page.content()
    condensed_html = condense_html(content)
    if len(condensed_html) > 250000:
        print("HTML content is too long. Truncating to 250,000 characters.")
        condensed_html = condensed_html[:250000]
    await async_openai_client.beta.assistants.update(
        assistant_id=assistant.id,
        instructions=ASSISTANT_INSTRUCTIONS + condensed_html,
    )
    return condensed_html


@cl.step(type="tool")
async def speech_to_text(audio_file):
    response = await async_openai_client.audio.transcriptions.create(
        model="whisper-1", file=audio_file
    )

    return response.text


async def upload_files(files: List[Element]):
    file_ids = []
    for file in files:
        uploaded_file = await async_openai_client.files.create(
            file=Path(file.path), purpose="assistants"
        )
        file_ids.append(uploaded_file.id)
    return file_ids


async def process_files(files: List[Element]):
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
    # Create a Thread
    thread = await async_openai_client.beta.threads.create()
    # Create a playwright session using browserless

    playwright = await async_api.async_playwright().start()
    async_browser = await playwright.chromium.connect_over_cdp(
        "ws://localhost:3000",
    )
    page = await async_browser.new_page()
    page.set_default_timeout(10000)
    cl.user_session.set("page", page)
    # Store thread ID in user session for later use
    cl.user_session.set("thread_id", thread.id)
    await cl.Avatar(name=assistant.name, path="./public/logo.png").send()
    await cl.Message(
        content=f"Hello, I'm {assistant.name}!", disable_feedback=True
    ).send()


@cl.on_message
async def main(message: cl.Message):
    thread_id = cl.user_session.get("thread_id")

    attachments = await process_files(message.elements)

    # Add a Message to the Thread
    oai_message = await async_openai_client.beta.threads.messages.create(
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
async def on_audio_chunk(chunk: cl.AudioChunk):
    if chunk.isStart:
        buffer = BytesIO()
        # This is required for whisper to recognize the file type
        buffer.name = f"input_audio.{chunk.mimeType.split('/')[1]}"
        # Initialize the session for a new audio stream
        cl.user_session.set("audio_buffer", buffer)
        cl.user_session.set("audio_mime_type", chunk.mimeType)

    # Write the chunks to a buffer and transcribe the whole audio at the end
    cl.user_session.get("audio_buffer").write(chunk.data)


@cl.step
async def screenshot():
    page = cl.user_session.get("page")
    current_step = cl.context.current_step
    # Simulate a running task
    # wait for network idle
    # await page.wait_for_load_state("networkidle")
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
            name="screenshot", output=str(e), parent_id=current_step.parent_id
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
