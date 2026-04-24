import pandas as pd
from langchain_core.tools import tool

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


#TOOL - 1
@tool
def doctor_specializations_available():
    '''
    Generates the list of all the specialized doctors.
    '''
    df = pd.read_csv(CSV_PATH)
    unique_specialization = list(df['specialization'].unique())
    return f'Here are the unique available doctors in our clinic {unique_specialization}'


#TOOL - 2
@tool
def get_doctor_by_specializations(specialization: str):
    '''
    Generates the doctors list who are available for a particular specialization
    '''
    df = pd.read_csv(CSV_PATH)
    doctors_available = list(df[df['specialization'].apply(norm) == norm(specialization)]['doctor_name'].unique())
    return f"Here are the list of doctors that are available for a particular {specialization} {doctors_available}"

#Tool-3
@tool
def doctors_available_timeslots(doctor_name: str, specialization: str):
    '''
    Generates the list of available date-slot for a doctor with a particular specialization
    '''
    df = pd.read_csv(CSV_PATH)
    filtered = df[
        (df['doctor_name'].apply(norm) == norm(doctor_name)) &
        (df['specialization'].apply(norm) == norm(specialization)) &
        (df['is_available'] == True)
    ]
    slot_availability = filtered['date_slot'].tolist()

    if len(slot_availability) == 0:
        return f"The {doctor_name} with following {specialization} is very busy and is not available."
    else:
        return f"The {doctor_name} with following {specialization} is available in the following timeslots {slot_availability}"


#Tool-4
@tool
def queue_position(patient_to_attend, doctor_name):
    '''
    Return the queue position of a patient for a particular doctor.
    Position = 1 means they are next in line.
    '''
    df = pd.read_csv(CSV_PATH)
    booked = df[
        (df['doctor_name'].apply(norm) == norm(doctor_name)) &
        (df['is_available'] == False)
    ].sort_values('date_slot').reset_index(drop=True)

    match = booked[booked['patient_to_attend'] == patient_to_attend]

    if match.empty:
        return f"Patient {patient_to_attend} is not found in {doctor_name}'s queue."

    position = match.index[0] + 1
    slot = match.iloc[0]['date_slot']

    return (
        f"Patient {patient_to_attend} is at position {position} "
        f"in {doctor_name}'s queue. Their scheduled slot is {slot}."
    )

SYSTEM_PROMPT = """


You are a helpful and professional medical clinic assistant. Your role is to help 
patients find the right doctor, check availability, and know their queue position.

You have access to the following tools — use them in order based on what the patient needs:

1. doctor_specializations_available — use this when a patient does not know what 
   specialization they need, or asks "what doctors do you have?".

2. get_doctor_by_specializations(specialization) — use this when a patient knows 
   their required specialization and wants to know which doctors are available.

3. doctors_available_timeslots(doctor_name, specialization) — use this when a 
   patient has chosen a doctor and wants to see open appointment slots.

4. queue_position(patient_to_attend, doctor_name) — use this when a patient provides 
   their patient ID and wants to know where they stand in a doctor's queue.

## Rules

- Always be polite, concise, and reassuring — patients may be anxious.
- Never guess or fabricate doctor names, specializations, or slot availability. 
  Only report what the tools return.
- If a patient's query is ambiguous, ask one clarifying question before calling a tool.
- If a tool returns no results, inform the patient clearly and suggest an alternative 
  (e.g., a different doctor or specialization).
- Do not expose raw data, IDs, or technical errors to the patient. Translate tool 
  output into natural, friendly language.
- Patient IDs are sensitive — do not repeat them back unnecessarily.


"""

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")


information_agent = create_agent(
    name="information_agent",
    model=llm,
    tools=[
        doctor_specializations_available,
        get_doctor_by_specializations,
        doctors_available_timeslots,
        queue_position,
    ],
    system_prompt=SYSTEM_PROMPT
)

