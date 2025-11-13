import os
import base64
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv


load_dotenv()

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

class NodePlan(BaseModel):
    nodes: list[dict] = Field(description="List of nodes to create, each with 'name' and 'description'")

def create_nodes_from_plan(user_message, file_refs=None):
    """Plan and create all nodes at once based on user input."""
    print("\n[PLANNING] Analyzing input and planning nodes...")

    # Create planning prompt
    plan_prompt = f"{user_message}\n\n[INTERNAL] Create a comprehensive plan of all financial nodes needed. Return JSON with structure: {{'nodes': [{{'name': 'node_name', 'description': 'node_desc'}}]}}"

    # Get the plan
    if file_refs:
        plan_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[*file_refs, plan_prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=NodePlan,
            )
        )
    else:
        plan_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=plan_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=NodePlan,
            )
        )

    # Parse and create nodes
    import json
    plan_data = json.loads(plan_response.text)
    nodes = plan_data.get("nodes", [])

    print(f"[PLAN COMPLETE] Creating {len(nodes)} nodes...\n")

    for node in nodes:
        print(f"[NODE CREATED] {node['name']}: {node['description']}")

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
print("  - Type 'plan:your request' to plan and create nodes")
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

    # Check if user wants to plan and create nodes
    if user_input.startswith('plan:'):
        request = user_input[5:].strip()
        nodes = create_nodes_from_plan(request, uploaded_files if uploaded_files else None)
        print(f"\n[COMPLETE] Created {len(nodes)} nodes based on your request.\n")
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