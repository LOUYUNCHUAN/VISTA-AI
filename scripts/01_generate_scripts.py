import asyncio
import json
import os
import random
import uuid
from typing import Literal, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from tqdm.asyncio import tqdm

# Load environment variables
load_dotenv()

# Configuration
TOTAL_DIALOGUES = 500
CLASSES = [
    "very_unsatisfied",
    "unsatisfied",
    "satisfied",
    "very_satisfied"
]
# Split distribution to reach exactly 450 Train / 50 Test (90%/10%)
DISTRIBUTION = {
    "very_unsatisfied": {"train": 113, "test": 12},
    "unsatisfied": {"train": 112, "test": 13},
    "satisfied": {"train": 113, "test": 12},
    "very_satisfied": {"train": 112, "test": 13},
}

DOMAINS = ["e-commerce", "telecom", "banking", "tech support"]
GENDERS = ["male", "female"]
AGE_GROUPS = ["young", "middle-aged", "elderly"]
CUSTOMER_ACCENTS = ["Singaporean English", "Indian English", "Chinese English", "Malay English"]
ENGINEER_ACCENTS = ["Standard American", "British", "Neutral Asian"]

# Pydantic schema for Structured Output
class CustomerProfile(BaseModel):
    gender: Literal["male", "female"]
    age_group: Literal["young", "middle-aged", "elderly"]
    accent: Literal["Singaporean English", "Indian English", "Chinese English", "Malay English"]

class EngineerProfile(BaseModel):
    gender: Literal["male", "female"]
    accent: str

class Turn(BaseModel):
    speaker: Literal["customer", "engineer"]
    text: str = Field(description="Turn text including paralinguistic tags like [sigh], [scoff], aggressive capitalization.")
    emotion_tag: str = Field(description="Emotion descriptor, e.g., 'explosive_anger', 'polite_neutral'.")
    tts_style_weight: float = Field(description="0.0 to 1.0. High intensity is 0.7-0.9, calm is 0.1-0.2.")
    offset_ms: int = Field(description="Inter-turn pause in ms. Negative for interruptions/cut-ins, positive for natural pauses.")

class Dialogue(BaseModel):
    dialogue_id: str
    target_duration_sec: int
    action_label: Literal["very_unsatisfied", "unsatisfied", "satisfied", "very_satisfied"]
    fine_grained_emotion: str
    split: Literal["train", "test"]
    domain: Literal["e-commerce", "telecom", "banking", "tech support"]
    customer_profile: CustomerProfile
    engineer_profile: EngineerProfile
    turns: List[Turn]

PROMPT_TEMPLATE = """You are an expert dialogue writer specializing in call-center linguistics, acoustic emotions, and regional Asian English accents.
Generate a multi-turn JSON dialogue script between a 'customer' and an 'engineer' based on the following parameters:

TARGET_CLASS: {action_label}
DOMAIN: {domain}
CUSTOMER_ACCENT: {accent}
TARGET_DURATION: {target_duration_sec} seconds (~60-75 words total)

Rules:
1. Ensure the text reflects the requested CUSTOMER_ACCENT (e.g., use subtle Singlish particles like 'lah', 'leh', 'meh' naturally for Singaporean English).
2. Inject paralinguistic cues in brackets (e.g., [sigh], [angry gasp], [scoff], [warm laugh]) and use ALL CAPS for shouting to guide the TTS engine.
3. Manage turn-taking dynamically via the 'offset_ms' field:
   - For 'urgent_follow_up' and 'at_risk_dissatisfied', the customer MUST interrupt the engineer at least once using a negative offset (e.g., -600 to -1000).
   - For 'standard_resolved' and 'promoter_delighted', use polite positive offsets (200 to 500).
4. The 'tts_style_weight' should map to the emotion: 0.7-0.9 for high intensity (rage/delight), 0.1-0.2 for neutral/calm.
5. Provide between 4 and 8 turns.
6. The exact 'dialogue_id' is {dialogue_id} and 'split' is {split}. Ensure they are exact in the output.
"""

async def generate_script(client: genai.Client, task_params: dict, max_retries=3):
    prompt = PROMPT_TEMPLATE.format(
        action_label=task_params["action_label"],
        domain=task_params["domain"],
        accent=task_params["customer_profile"]["accent"],
        target_duration_sec=task_params["target_duration_sec"],
        dialogue_id=task_params["dialogue_id"],
        split=task_params["split"]
    )
    
    for attempt in range(max_retries):
        try:
            # We use gemini-3.6-flash as recommended by the API
            response = await client.aio.models.generate_content(
                model='gemini-3.6-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=Dialogue,
                    temperature=0.7
                ),
            )
            dialogue_json = response.text
            return json.loads(dialogue_json)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to generate script {task_params['dialogue_id']} after {max_retries} attempts: {e}")
                return None
            await asyncio.sleep(2 ** attempt)

async def main():
    random.seed(42)
    os.makedirs("data/raw", exist_ok=True)
    output_file = "data/raw/dialogues.jsonl"
    
    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY not found in environment.")
        return

    client = genai.Client()
    
    # Generate task parameters
    tasks = []
    dial_counter = 1
    
    for action_label, splits in DISTRIBUTION.items():
        for split, count in splits.items():
            for _ in range(count):
                task = {
                    "dialogue_id": f"dial_{dial_counter:03d}_{action_label[:5]}",
                    "action_label": action_label,
                    "split": split,
                    "domain": random.choice(DOMAINS),
                    "target_duration_sec": random.randint(25, 35),
                    "customer_profile": {
                        "gender": random.choice(GENDERS),
                        "age_group": random.choice(AGE_GROUPS),
                        "accent": random.choice(CUSTOMER_ACCENTS)
                    },
                    "engineer_profile": {
                        "gender": random.choice(GENDERS),
                        "accent": random.choice(ENGINEER_ACCENTS)
                    }
                }
                tasks.append(task)
                dial_counter += 1
                
    random.shuffle(tasks) # Shuffle so we don't do all of one class at once
    
    print(f"Starting generation of {len(tasks)} dialogues...")
    
    # We will process in batches to avoid rate limits
    BATCH_SIZE = 10
    all_results = []
    
    for i in tqdm(range(0, len(tasks), BATCH_SIZE), desc="Batches"):
        batch_tasks = tasks[i:i+BATCH_SIZE]
        coroutines = [generate_script(client, t) for t in batch_tasks]
        batch_results = await asyncio.gather(*coroutines)
        
        # Save incrementally
        with open(output_file, "a") as f:
            for res in batch_results:
                if res:
                    f.write(json.dumps(res) + "\n")
                    all_results.append(res)
                    
        # Small delay to respect rate limits
        await asyncio.sleep(2)
        
    print(f"Finished! Successfully generated {len(all_results)}/{TOTAL_DIALOGUES} dialogues.")
    print(f"Output saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
