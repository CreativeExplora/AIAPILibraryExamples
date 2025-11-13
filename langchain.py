import os
from typing import TypedDict, Any
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.agents import create_agent, AgentState
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.middleware import AgentMiddleware,wrap_model_call, ModelRequest, ModelResponse, wrap_tool_call, dynamic_prompt 

# Load environment variables from .env file
load_dotenv()

# Read Gemini API key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

basic_model = "gemini-2.0-flash"
advanced_model = "gemini-2.5-flash"

class WeatherInfo(BaseModel):
    city: str
    temperature: str
    condition: str


class CustomState(AgentState):
    user_role: str
    user_preferences: dict


@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """Generate system prompt based on user role."""
    user_role = request.state.get("user_role", "user")
    base_prompt = "You are a helpful assistant. You have access to the following tools:\n- get_weather: Get weather for a given city."

    if user_role == "expert":
        return f"{base_prompt} Provide detailed technical responses."
    elif user_role == "beginner":
        return f"{base_prompt} Explain concepts simply and avoid jargon."

    return base_prompt

@wrap_model_call
def model_selector(request: ModelRequest, handler) -> ModelResponse:
    message_count = len(request.state["messages"])
    if message_count >= 20:
        selected_model_name = advanced_model
    else:
        selected_model_name = basic_model
    # Create a new model instance with the selected model name
    request.model = ChatGoogleGenerativeAI(
        model=selected_model_name,
        google_api_key=GEMINI_API_KEY,
    )
    return handler(request)

@wrap_tool_call
def exception_handler(request, handler) -> ToolMessage:
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

@tool
def get_forecast(city: str) -> str:
    """Get forecast for a given city."""
    return f"The forecast for {city} is sunny all week!"

class CustomMiddleware(AgentMiddleware):
    state_schema = CustomState
    tools = [get_weather, get_forecast]

    def before_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
        """Process state before model invocation."""
        # Access custom state fields
        user_prefs = state.get("user_preferences", {})
        user_role = state.get("user_role", "user")

        # You can modify the state or return additional context
        print(f"User preferences: {user_prefs}")
        print(f"User role: {user_role}")

        return None

# Initialize the Gemini model with API key
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
)

agent = create_agent(
    model=model,
    tools=[get_weather, get_forecast],
    middleware=[model_selector, exception_handler, user_role_prompt],
    # response_format=ToolStrategy(WeatherInfo),  # Commented out to avoid recursion in streaming
    state_schema=CustomState
)

# Run the agent
# The agent can now track additional state beyond messages
result = agent.invoke({
    "messages": [{"role": "user", "content": "I prefer technical explanations. What is the weather in sf?"}],
    "user_role": "expert",
    "user_preferences": {"style": "technical", "verbosity": "detailed"},
})
# Print the result
print(result["messages"][-1].content)

print("\n" + "="*80)
print("Streaming Example:")
print("="*80 + "\n")

# Stream agent execution
for chunk in agent.stream({
    "messages": [{"role": "user", "content": "What is the weather and forecast in Tokyo?"}],
    "user_role": "beginner",
    "user_preferences": {"style": "simple", "verbosity": "brief"},
}, stream_mode="values"):
    # Each chunk contains the full state at that point
    latest_message = chunk["messages"][-1]
    if latest_message.content:
        print(f"Agent: {latest_message.content}")
    elif hasattr(latest_message, 'tool_calls') and latest_message.tool_calls:
        print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")