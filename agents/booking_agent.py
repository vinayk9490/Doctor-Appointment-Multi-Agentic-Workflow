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
def appointment_booking(patient_id, specialization, doctor_name, date_slot: str = None):
    '''
    Books an appointment for a patient with a particular doctor and specialization.
    If date_slot is provided (format: YYYY-MM-DD HH:MM:SS), books that specific slot.
    If date_slot is omitted, books the earliest available slot.
    '''
    df = pd.read_csv(CSV_PATH)

    mask = (
        (df["doctor_name"].apply(norm) == norm(doctor_name)) &
        (df["specialization"].apply(norm) == norm(specialization)) &
        (df["is_available"] == True)
    )

    if date_slot:
        date_slot = parse_date(date_slot)
        mask = mask & (df["date_slot"] == date_slot)

    booking = df[mask]

    if len(booking) == 0:
        if date_slot:
            return (
                f"The slot {date_slot} with Dr. {doctor_name} ({specialization}) is not available. "
                "Please ask the patient to choose a different slot."
            )
        return f"Dr. {doctor_name} ({specialization}) has no available slots right now. Please try again later."

    idx = booking.index[0]
    df.loc[idx, 'is_available'] = False
    df.loc[idx, 'patient_to_attend'] = float(patient_id)
    df.to_csv(CSV_PATH, index=False)

    appointment_date = df.loc[idx, 'date_slot']
    return f"Appointment confirmed! Dr. {doctor_name} ({specialization}) on {appointment_date} for patient {patient_id}."


# Tool - 2
@tool
def view_all_booked_appointments(patient_id):
    '''
    Returns all booked appointments for a particular patient.
    '''
    df = pd.read_csv(CSV_PATH)
    appointments = df[df['patient_to_attend'] == float(patient_id)][['doctor_name', 'specialization', 'date_slot']]

    if len(appointments) == 0:
        return f"Patient {patient_id} does not have any bookings scheduled with our dental clinic."

    return f"Patient {patient_id} has the following appointments scheduled: {appointments.to_dict('records')}"


# Tool - 3
@tool
def view_appointments_with_specialization(patient_id, specialization):
    '''
    Returns all appointments a patient has for a particular specialization.
    '''
    df = pd.read_csv(CSV_PATH)
    specific_appointments = df[
        (df["specialization"].apply(norm) == norm(specialization)) &
        (df["patient_to_attend"] == float(patient_id))
    ][['doctor_name', 'specialization', 'date_slot']]

    if len(specific_appointments) == 0:
        return f"Patient {patient_id} has no appointments scheduled for {specialization}."

    return f"Patient {patient_id} has the following {specialization} appointments: {specific_appointments.to_dict('records')}"


BOOKING_SYSTEM_PROMPT = """
You are a clinic booking assistant. Your job is to call tools — not to describe what you will do.

CRITICAL RULE: Never say "I'll book", "I will schedule", or any future-tense promise.
Call the tool immediately. Only respond to the patient AFTER the tool returns a result.

You have access to the following tools:

1. appointment_booking(patient_id, specialization, doctor_name, date_slot=None)
   - Call this AS SOON AS you have patient_id, doctor_name, and specialization.
   - If the patient gives a date/time, convert it to YYYY-MM-DD HH:MM:SS and pass as date_slot.
     Examples: "July 8 2026 8:00 AM" → "2026-07-08 08:00:00"
   - If date_slot is not given, omit it — the earliest available slot will be booked.
   - If specialization is not in the current message, check conversation history.
     If still unknown, ask ONE question: "What specialization is Dr. X?"

2. view_all_booked_appointments(patient_id) — call this to show all appointments for a patient.

3. view_appointments_with_specialization(patient_id, specialization) — call this to show
   appointments for a specific specialization.

## Behavior
- When you have enough information → call the tool immediately, no preamble.
- Only tell the patient the outcome AFTER the tool responds.
- If the tool returns "Appointment confirmed!" → tell the patient it's booked.
- If the tool returns "not available" → tell the patient and suggest alternatives.
- Patient IDs are sensitive — do not repeat them back unnecessarily.
"""


from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")

booking_agent = create_agent(
    name="booking_agent",
    model=llm,
    tools=[
        appointment_booking,
        view_all_booked_appointments,
        view_appointments_with_specialization,
    ],
    system_prompt=BOOKING_SYSTEM_PROMPT
)
