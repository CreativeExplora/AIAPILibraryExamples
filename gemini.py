import os
import base64
import json
from datetime import datetime
from typing import Optional
from uuid import uuid4
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv


load_dotenv()
nodes_table = []

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

FINANCIAL_SYSTEM_PROMPT = """YOU ARE A HIGHLY ADVANCED, NEXT GENERATION FINANCIAL MODELING ASSISTANT FOR A FINANCIAL MODELING, MANAGEMENT, AND ANALYSIS TOOL.

**You currently only have access to the canvas interface. This is where users can create nodes which are group (or single) financial transactions. Nodes can be connected to form paths, users can select a start and end node and the engine will calculate the aggregate effect of the nodes on the financial statements.**

CORE BEHAVIOR: Proactively create nodes to model business scenarios. Don't just explain - ACTUALLY MODEL IT.

MODEL ALL DETAILS OF THE USER INPUT EVEN IF IT MEANS CREATING HUNDREDS OF NODES.

If a user tells you to create a model, you should first create the nodes and then create the edges.
If the user uploads a long business strategy plan or such, first create the nodes, then create the edges, then create the variables that are used in those nodes.
The model should match the complexity of the user's input. For example, if the user uploads a long business strategy plan, you should create a lot of nodes and edges, possibly 50 or even over 100 nodes and edges.

"""

client = genai.Client()

class AccountEntry(BaseModel):
    """A single debit or credit entry."""
    amount: str = Field(description="Amount for this entry")
    account: str = Field(description="Account name")

class Transaction(BaseModel):
    """Transaction details for a financial node."""
    name: str = Field(description="Name of the transaction")
    debits: list[AccountEntry] = Field(description="List of debit entries")
    credits: list[AccountEntry] = Field(description="List of credit entries")

class Node(BaseModel):
    """Financial transaction node model matching database schema."""
    node_name: str = Field(description="Name of the financial node")
    constraints: Optional[list[str]] = Field(default=None, description="List of constraint strings for the node")
    transaction: Optional[list[Transaction]] = Field(default=None, description="List of transactions for this node")
    transaction_description: Optional[str] = Field(default=None, description="Description of the transaction")
    absolute_start_utc: str = Field(description="Absolute start timestamp in UTC (ISO format)")
    absolute_end_utc: Optional[str] = Field(default=None, description="Absolute end timestamp in UTC (ISO format)")
    start_offset_rule: Optional[str] = Field(default=None, description="Rule for start offset")
    end_offset_rule: Optional[str] = Field(default=None, description="Rule for end offset")
    recurrence_rule: Optional[str] = Field(default=None, description="Recurrence rule for repeating transactions")
    expected_value: float = Field(default=0, description="Expected numeric value")

def create_nodes(user_message, file_refs=None):
    """Generate nodes using structured output."""
    print("\n[GENERATING NODES]...")

    # Get structured output
    if file_refs:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[*file_refs, user_message],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=list[Node],
            )
        )
    else:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_message,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=list[Node],
            )
        )

    # Parse nodes
    nodes = json.loads(response.text)

    print(f"\n[CREATED] {len(nodes)} nodes:\n")

    for idx, node in enumerate(nodes, 1):
        nodes_table.append(node)
        print(f"--- Node {idx} ---")
        print(f"node_name: {node['node_name']}")
        print(f"constraints: {node.get('constraints')}")
        print(f"transaction: {node.get('transaction')}")
        print(f"transaction_description: {node.get('transaction_description')}")
        print(f"absolute_start_utc: {node['absolute_start_utc']}")
        print(f"absolute_end_utc: {node.get('absolute_end_utc')}")
        print(f"start_offset_rule: {node.get('start_offset_rule')}")
        print(f"end_offset_rule: {node.get('end_offset_rule')}")
        print(f"recurrence_rule: {node.get('recurrence_rule')}")
        print(f"expected_value: {node.get('expected_value', 0)}")
        print()

    return nodes

chat = client.chats.create(
    model="gemini-2.5-flash",
    config=types.GenerateContentConfig(
        temperature=0.5,
        system_instruction=FINANCIAL_SYSTEM_PROMPT,
    )
)

def upload_file_from_path(file_path):
    """Upload a local file to the Gemini File API."""
    try:
        with open(file_path, 'rb') as f:
            mime_type = None
            if file_path.endswith('.pdf'):
                mime_type = 'application/pdf'
            elif file_path.endswith(('.png', '.jpg', '.jpeg')):
                mime_type = f'image/{file_path.split(".")[-1]}'

            uploaded_file = client.files.upload(
                file=f,
                config=dict(mime_type=mime_type) if mime_type else {}
            )
        return uploaded_file
    except Exception as e:
        print(f"Error uploading file from path: {e}")
        return None

print("Chat started. Commands:")
print("  - Type 'upload:/path/to/file' to upload a local file")
print("  - Type 'create:your request' to plan and create nodes")
print("  - Type 'quit' to exit\n")

uploaded_files = []  # Track uploaded files across conversation

while True:
    user_input = input("You: ")

    if user_input.lower() == 'quit':
        break

    # Check if user wants to upload files
    if user_input.startswith('upload:'):
        file_refs = []
        file_paths = user_input[7:].strip().split(',')

        for file_path in file_paths:
            file_path = file_path.strip()
            uploaded = upload_file_from_path(file_path)

            if uploaded:
                file_refs.append(uploaded)
                uploaded_files.append(uploaded)
                print(f"Uploaded: {file_path}")

        if not file_refs:
            print("No files were uploaded successfully.")
        continue

    # Check if user wants to create nodes
    if user_input.startswith('create:'):
        request = user_input[7:].strip()
        nodes = create_nodes(request, uploaded_files if uploaded_files else None)
        print(f"\n[COMPLETE] {len(nodes)} nodes created.\n")
        continue

    # Normal chat message
    if uploaded_files:
        response = chat.send_message_stream([*uploaded_files, user_input])
    else:
        response = chat.send_message_stream(user_input)

    print("Agent: ", end="", flush=True)
    for chunk in response:
        # Handle thought signatures
        if hasattr(chunk, 'candidates') and chunk.candidates:
            for candidate in chunk.candidates:
                if hasattr(candidate, 'content') and candidate.content:
                    for part in candidate.content.parts:
                        if hasattr(part, 'thought_signature') and part.thought_signature:
                            print(f"\n[THINKING]: {base64.b64encode(part.thought_signature).decode('utf-8')}\n", end="", flush=True)

        # Handle text content
        if chunk.text:
            print(chunk.text, end="", flush=True)
    print()  # Newline after streaming completes