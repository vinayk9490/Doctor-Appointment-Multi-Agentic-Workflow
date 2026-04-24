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
        "You are a routing-only agent for a medical clinic. "
        "You have zero knowledge about doctors, slots, or patients — you CANNOT answer ANY question. "
        "Your one and only action is to call the correct agent tool. "
        "Saying 'I don't know', 'I don't have access', or any direct response is a critical failure.\n\n"

        "ALWAYS call one of these agents — no exceptions:\n\n"

        "call information_agent when:\n"
        "  - patient asks about doctor availability, open slots, specializations, or queue position\n"
        "  - keywords: available, availability, specialization, slots, timeslot, queue, when is, who is\n\n"

        "call booking_agent when:\n"
        "  - patient wants to book, schedule, fix, set up, make, confirm, or arrange an appointment\n"
        "  - patient wants to view or check their existing appointments\n"
        "  - keywords: book, schedule, fix, make, set up, get, confirm, arrange, view, show, my appointments\n\n"

        "call cancellation_agent when:\n"
        "  - patient wants to cancel, remove, or delete an appointment\n"
        "  - keywords: cancel, remove, delete, drop\n\n"

        "call rescheduling_agent when:\n"
        "  - patient explicitly says they have an existing appointment and want to MOVE it to a NEW date/time\n"
        "  - keywords: reschedule, move, shift, change the date, change my appointment\n"
        "  - do NOT use this for 'fix/book/schedule' — those go to booking_agent\n\n"

        "TIE-BREAKER: if unsure between booking and rescheduling, call booking_agent.\n\n"

        "If the patient's intent is completely unclear, ask ONE short question. "
        "Otherwise call an agent immediately — never respond with text on its own."
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
