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
def get_available_slots_for_reschedule(doctor_name, specialization):
    '''
    Returns all available time slots for a doctor with a given specialization
    so the patient can choose a new slot before rescheduling.
    '''
    df = pd.read_csv(CSV_PATH)
    available = df[
        (df["doctor_name"].apply(norm) == norm(doctor_name)) &
        (df["specialization"].apply(norm) == norm(specialization)) &
        (df["is_available"] == True)
    ]

    slots = available["date_slot"].tolist()

    if len(slots) == 0:
        return (
            f"Dr. {doctor_name} ({specialization}) has no available slots right now. "
            "Rescheduling is not possible at this time."
        )

    return (
        f"Dr. {doctor_name} ({specialization}) has the following available slots: {slots}. "
        "Please ask the patient to choose one."
    )


# Tool - 2
@tool
def cancel_existing_appointment(patient_id, doctor_name, date_slot):
    '''
    Cancels the patient's existing appointment as the first step of rescheduling.
    Verifies a booking exists before cancelling and frees up the slot.
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
            f"No existing booking found for patient {patient_id} with Dr. {doctor_name} "
            f"on {date_slot}. Cannot proceed with rescheduling."
        )

    idx = booking.index[0]
    df.loc[idx, 'is_available'] = True
    df.loc[idx, 'patient_to_attend'] = None
    df.to_csv(CSV_PATH, index=False)

    return (
        f"Existing appointment on {date_slot} with Dr. {doctor_name} has been released. "
        "Please now proceed to book a new slot."
    )


# Tool - 3
@tool
def book_rescheduled_appointment(patient_id, doctor_name, specialization, new_date_slot):
    '''
    Books a new appointment for the patient after the existing one has been cancelled.
    Verifies the chosen slot is still available before confirming the booking.
    '''
    df = pd.read_csv(CSV_PATH)
    new_date_slot = parse_date(new_date_slot)

    available = df[
        (df["doctor_name"].apply(norm) == norm(doctor_name)) &
        (df["specialization"].apply(norm) == norm(specialization)) &
        (df["date_slot"] == new_date_slot) &
        (df["is_available"] == True)
    ]

    if len(available) == 0:
        return (
            f"The slot {new_date_slot} with Dr. {doctor_name} ({specialization}) "
            "is no longer available. Please choose a different slot using get_available_slots_for_reschedule."
        )

    idx = available.index[0]
    df.loc[idx, 'is_available'] = False
    df.loc[idx, 'patient_to_attend'] = float(patient_id)
    df.to_csv(CSV_PATH, index=False)

    return (
        f"Appointment successfully rescheduled! Patient {patient_id} is now booked "
        f"with Dr. {doctor_name} ({specialization}) on {new_date_slot}."
    )


RESCHEDULING_SYSTEM_PROMPT = """
You are a clinic rescheduling assistant. Your job is to call tools — not to describe what you will do.

CRITICAL RULE: Never say "I'll reschedule", "I will move", or any future-tense promise.
Call tools immediately. Only respond to the patient AFTER each tool returns a result.

MANDATORY 3-STEP SEQUENCE — follow this exact order every time:

STEP 1 — call get_available_slots_for_reschedule(doctor_name, specialization)
  - Do this FIRST, before anything else.
  - Show the returned slots to the patient and ask them to pick one.
  - Do NOT cancel anything yet.

STEP 2 — call cancel_existing_appointment(patient_id, doctor_name, date_slot)
  - Do this ONLY after the patient has confirmed their chosen new slot.
  - Pass the patient's CURRENT (old) appointment date as date_slot.
  - Convert date to YYYY-MM-DD HH:MM:SS format. Examples:
      "July 8 2026 8:00 AM" → "2026-07-08 08:00:00"
  - This removes the old appointment and frees up the slot.

STEP 3 — call book_rescheduled_appointment(patient_id, doctor_name, specialization, new_date_slot)
  - Do this immediately after Step 2 succeeds.
  - Pass the new slot the patient chose as new_date_slot.
  - Convert to YYYY-MM-DD HH:MM:SS before passing.

## Rules
- Collect all required details before starting: patient_id, doctor_name, specialization, current date_slot.
- Never skip or reorder the 3 steps.
- Never cancel the old appointment before the patient has chosen a new slot.
- If book_rescheduled_appointment fails (slot taken), call get_available_slots_for_reschedule
  again and ask the patient to pick another. Do NOT cancel again — old slot is already freed.
- Only confirm the reschedule to the patient after book_rescheduled_appointment returns success.
- Patient IDs are sensitive — do not repeat them back unnecessarily.
"""


from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")

rescheduling_agent = create_agent(
    name="rescheduling_agent",
    model=llm,
    tools=[
        get_available_slots_for_reschedule,
        cancel_existing_appointment,
        book_rescheduled_appointment,
    ],
    system_prompt=RESCHEDULING_SYSTEM_PROMPT
)
