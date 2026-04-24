import pandas as pd
from langchain_core.tools import tool
from langchain.agents import create_agent

CSV_PATH = r"data artifacts/doctor_availability.csv"

def norm(s) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    if s.startswith("dr."):
        s = s[3:].strip()
    elif s.startswith("dr "):
        s = s[3:].strip()
    return s.replace(" ", "_")

def parse_date(date_slot: str) -> str:
    try:
        return pd.to_datetime(date_slot).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return date_slot


# Tool - 1
@tool
def cancel_appointment(patient_id, doctor_name, date_slot):
    '''
    Cancels an existing appointment for a patient with a particular doctor on a given date slot.
    Before cancellation, verifies that a booking actually exists for the patient.
    '''
    df = pd.read_csv(CSV_PATH)
    date_slot = parse_date(date_slot)

    booking = df[
        (df["patient_to_attend"] == float(patient_id)) &
        (df["doctor_name"].apply(norm) == norm(doctor_name)) &
        (df["date_slot"] == date_slot) &
        (df["is_available"] == False)
    ]

    if len(booking) == 0:
        return (
            f"No booking found for patient {patient_id} with Dr. {doctor_name} "
            f"on {date_slot}. Nothing to cancel."
        )

    idx = booking.index[0]
    df.loc[idx, 'is_available'] = True
    df.loc[idx, 'patient_to_attend'] = None
    df.to_csv(CSV_PATH, index=False)

    return (
        f"Appointment successfully cancelled for patient {patient_id} "
        f"with Dr. {doctor_name} on {date_slot}."
    )


CANCELLATION_SYSTEM_PROMPT = """
You are a clinic cancellation assistant. Your only role is to help patients cancel
their existing appointments.

You have access to the following tool:

1. cancel_appointment(patient_id, doctor_name, date_slot) — use this when a patient
   wants to cancel a specific appointment. The tool first verifies the booking exists
   before proceeding with cancellation.

## Rules
- Always confirm the patient_id, doctor_name, and date_slot before calling cancel_appointment.
- If any of the three details are missing, ask for them before proceeding.
- If the tool reports no booking found, inform the patient politely and suggest they
  check their appointment details.
- Never cancel an appointment without all three confirmed details.
- Keep responses short, clear, and empathetic.
- Patient IDs are sensitive — do not repeat them back unnecessarily.
"""


from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")

cancellation_agent = create_agent(
    name="cancellation_agent",
    model=llm,
    tools=[cancel_appointment],
    system_prompt=CANCELLATION_SYSTEM_PROMPT
)
