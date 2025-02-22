import os
import asyncio

from typing import Sequence

from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent, CodeExecutorAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination

from autogen_ext.agents.web_surfer import MultimodalWebSurfer

from autogen_agentchat.teams import SelectorGroupChat, MagenticOneGroupChat, RoundRobinGroupChat
from autogen_agentchat.ui import Console

from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.tools.code_execution import PythonCodeExecutionTool

from dotenv import load_dotenv
from pathlib import Path


async def main() -> None:
    # Load environment variables from .env file
    print("Loading environment variables...")
    # Construct the path to the .env file at the project root
    env_path = Path(__file__).resolve().parents[3] / ".env"
    load_dotenv(dotenv_path=env_path)
    
    # Create a model client using AzureOpenAIChatCompletionClient and the information from the .env file
    model_client = AzureOpenAIChatCompletionClient(
        model=os.getenv("AZURE_OPENAI_MODEL"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )

    # Initialize the web surfer agent
    web_agent = MultimodalWebSurfer(
        name="web_agent",
        model_client=model_client,
        headless=False,
        to_save_screenshots=True,
        use_ocr=False,
        # browser data directory is in the same directory as the script
        debug_dir=os.path.join(os.path.dirname(__file__), "screenshots"),
        browser_data_dir=os.path.join(os.path.dirname(__file__), "browser_data"),
        downloads_folder=os.path.join(os.path.dirname(__file__), "browser_downloads"),
        start_page=None
    )

    # initialize assistant agent with codeexecutor agent
    tool = PythonCodeExecutionTool(LocalCommandLineCodeExecutor(work_dir="coding"))
    assistant_coder = AssistantAgent(
        name="assistant", 
        model_client=model_client,
        tools=[tool],
        description="An agent that helps summarize information and prepare for distribution using Python code execution.",
        system_message="""You are an AI assistant .
        """,
        model_client_stream=True,
        reflect_on_tool_use=True
    )
    
    # Initialize the coder agent with a code execution tool
    code_executor_tool = PythonCodeExecutionTool(LocalCommandLineCodeExecutor(work_dir="coding"))
    coder_agent = AssistantAgent(
        name="coder_agent",
        model_client=model_client,
        tools=[code_executor_tool],
        description="An agent that executes Python code.",
        system_message="""You are a coder agent that executes Python code.""",
        model_client_stream=True,
        reflect_on_tool_use=True
    )

    # Create a user proxy agent
    user_proxy = UserProxyAgent("User", input_func=input)

    # Create the termination condition which will end the conversation when the user says "APPROVE".
    termination = TextMentionTermination("exit")

    # Create magentic one group chat with the assistant, web_agent and user proxy agents
    
    team = MagenticOneGroupChat(    
        participants=[web_agent, user_proxy, assistant_coder, coder_agent],
        model_client=model_client,
        termination_condition=termination,
        max_turns=20,
        max_stalls=3,
        final_answer_prompt="Review the information and provide a final answer in simple markdown."
    )
    
    
    user_message = """
    go to https://techcommunity.microsoft.com/blog/educatordeveloperblog/docaider-automated-documentation-maintenance-for-open-source-github-repositories/4245588 

    Collect short description, features and urls for me to insert into meeting minutes and output as simple markdown but richly formatted and with non-cringe emoji.

    ask the user (userproxy agent) if the web_surfer agent indicates some action required for cookie prompts, login credentials, etc.

    Ask the user whether to proceed if you suspect there is no more progress to be made. When everyone agrees the report should be finalzed, ask the use the assistant_coder to save the markdown to PDF with relevant filename and yy-MM-dd HHmm - DocumentName format.
    """

    user_message = """
    navigate directly to these urls
    -  http://f5uh4.sdcmlookup.com/rd/4eLQvQ4580FwAA85yqaeczsiyr263KDAALFZBCXAUNQX173813OWSQ17388G12
    - https://tecnichenuove.mn-ssl.com/nl/link?c=geda&d=5a43&h=17fsp9960bb1p0f7c1r678c5up&i=3og&iw=1&p=T113145259&s=lp&sn=2k4f&z=17p55
    both are suspected of being malicious. analyze the content including any possible hidden tracking pixels or strange behavior? do not search only attempt to access this url directly.

    it's possible the links are already dead, this is to be expected. But if you do find these links are still active, please provide a summary of the content and any suspicious behavior.

    if you get stuck escalate to the human in the loop.
     
    """

    user_message = """
    immediately ask the user (userproxy agent) for what assistance they require before starting any work. do not ask other agents.
    """
    await Console(team.run_stream(task=user_message))

asyncio.run(main())