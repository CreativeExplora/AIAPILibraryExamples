import os
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

chat= client.chats.create(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            temperature=0.5,
            system_instruction=FINANCIAL_SYSTEM_PROMPT)
    )

def plan_nodes_to_create() -> list[str]:
    return chat.send_message_stream(["[INTERNAL USE ONLY] Plan the nodes to create based on the user's input."], 
        config={
            "response_mime_type": "application/json",
            "response_json_schema": list[str],
        }
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
print("  - Type 'quit' to exit\n")

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
                print(f"Uploaded: {file_path}")

        if file_refs:
            print("Files uploaded. Ask a question about them:")
            follow_up = input("You: ")
            # Send message with uploaded files
            response = chat.send_message_stream([*file_refs, follow_up])
        else:
            print("No files were uploaded successfully.")
            continue
    else:
        # Normal message
        response = chat.send_message_stream(user_input)

    print("Agent: ", end="", flush=True)
    for chunk in response:
        if chunk.text:
            print(chunk.text, end="", flush=True)
    print()  # Newline after streaming completes