from dotenv import load_dotenv
load_dotenv()

import os
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = os.getenv('LANGCHAIN_TRACING_V2', 'false')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT', 'default')

from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor

from agents.information_agent import information_agent
from agents.booking_agent import booking_agent
from agents.cancellation_agent import cancellation_agent
from agents.rescheduling_agent import rescheduling_agent

model = ChatOpenAI(model="gpt-4o")

workflow = create_supervisor(
    [information_agent, booking_agent, cancellation_agent, rescheduling_agent],
    model=model,
    prompt=(
        "You are a router for a medical clinic. Your ONLY job is to read the patient's intent "
        "and immediately call the correct agent. You must NEVER answer any question yourself, "
        "NEVER add commentary, and NEVER perform any action — only route.\n\n"

        "ROUTING RULES — match the patient's intent to exactly one agent:\n\n"

        "→ information_agent\n"
        "  Use when the patient asks about: what specializations exist, which doctors are available, "
        "open time slots for a doctor, or their position in a doctor's queue.\n"
        "  Trigger words: 'available', 'availability', 'specialization', 'slots', 'queue', 'when is', 'who is'.\n\n"

        "→ booking_agent\n"
        "  Use when the patient wants to CREATE a new appointment OR view their existing appointments.\n"
        "  Trigger words: 'book', 'schedule', 'fix', 'set up', 'make', 'get', 'confirm', 'arrange', 'view appointments', 'show appointments'.\n"
        "  IMPORTANT: 'fix an appointment' and 'schedule an appointment' both mean BOOK NEW — route to booking_agent.\n\n"

        "→ cancellation_agent\n"
        "  Use when the patient wants to CANCEL (permanently remove) an existing appointment.\n"
        "  Trigger words: 'cancel', 'remove', 'delete', 'drop'.\n\n"

        "→ rescheduling_agent\n"
        "  Use ONLY when the patient explicitly says they have an EXISTING appointment and want to MOVE it to a different date or time.\n"
        "  Trigger words: 'reschedule', 'move', 'change the date', 'shift', 'change my appointment'.\n"
        "  IMPORTANT: Do NOT route here unless the patient clearly says they want to change an existing booking.\n\n"

        "DISAMBIGUATION RULE:\n"
        "  - 'I want to book/schedule/fix an appointment' → booking_agent (new booking)\n"
        "  - 'I want to reschedule/move my appointment' → rescheduling_agent (change existing)\n"
        "  - When in doubt between booking and rescheduling, always choose booking_agent.\n\n"

        "If the intent cannot be determined even after applying all rules above, ask exactly ONE short clarifying question."
    )
)

app = workflow.compile()

# Save workflow diagram as PNG
png_data = app.get_graph().draw_mermaid_png()
with open("workflow_diagram.png", "wb") as f:
    f.write(png_data)

# Chatbot loop
print("Clinic Assistant: Hello! How can I help you today? (type 'exit' to quit)")

conversation_history = []

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ("exit", "quit", "bye"):
        print("Clinic Assistant: Thank you for visiting. Goodbye!")
        break

    if not user_input:
        continue

    conversation_history.append({"role": "user", "content": user_input})

    result = app.invoke({"messages": conversation_history})

    # Pick the last sub-agent message — supervisor wraps with a meta-response
    # that doesn't contain actual data, so we look for the last named agent message
    sub_agents = {"information_agent", "booking_agent", "cancellation_agent", "rescheduling_agent"}
    sub_agent_messages = [
        m for m in result["messages"]
        if getattr(m, "name", None) in sub_agents
        and m.content
        and "transferring back" not in m.content.lower()
    ]

    assistant_message = sub_agent_messages[-1].content if sub_agent_messages else result["messages"][-1].content
    conversation_history.append({"role": "assistant", "content": assistant_message})

    print(f"Clinic Assistant: {assistant_message}")
