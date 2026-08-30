import json
import random
import math
import uuid

input_file = "data/raw/dialogues.jsonl"
target_total = 500

# Load existing dialogues
existing_dialogues = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            existing_dialogues.append(json.loads(line))

current_count = len(existing_dialogues)
if current_count == 0:
    print("No existing dialogues to augment.")
    exit()

needed = target_total - current_count
print(f"Found {current_count} base dialogues. Augmenting to create {needed} more...")

DOMAINS = ["e-commerce", "telecom", "banking", "tech support"]
GENDERS = ["male", "female"]
AGE_GROUPS = ["young", "middle-aged", "elderly"]
CUSTOMER_ACCENTS = ["Singaporean English", "Indian English", "Chinese English", "Malay English"]
ENGINEER_ACCENTS = ["Standard American", "British", "Neutral Asian"]

new_dialogues = []

# To ensure perfectly balanced classes (125 each)
CANONICAL_MAP = {
    "urgent_follow_up": "very_unsatisfied",
    "at_risk_dissatisfied": "unsatisfied",
    "standard_resolved": "satisfied",
    "promoter_delighted": "very_satisfied",
    "very_unsatisfied": "very_unsatisfied",
    "unsatisfied": "unsatisfied",
    "satisfied": "satisfied",
    "very_satisfied": "very_satisfied"
}

class_counts = {
    "very_unsatisfied": sum(1 for d in existing_dialogues if CANONICAL_MAP.get(d["action_label"]) == "very_unsatisfied"),
    "unsatisfied": sum(1 for d in existing_dialogues if CANONICAL_MAP.get(d["action_label"]) == "unsatisfied"),
    "satisfied": sum(1 for d in existing_dialogues if CANONICAL_MAP.get(d["action_label"]) == "satisfied"),
    "very_satisfied": sum(1 for d in existing_dialogues if CANONICAL_MAP.get(d["action_label"]) == "very_satisfied"),
}

for action_label in class_counts.keys():
    target_for_class = 125
    needed_for_class = max(0, target_for_class - class_counts[action_label])
    
    # Get all base dialogues for this class
    base_samples = [d for d in existing_dialogues if CANONICAL_MAP.get(d["action_label"]) == action_label]
    
    for i in range(needed_for_class):
        # Pick a random base sample to augment
        base = random.choice(base_samples)
        
        # Deep copy
        new_d = json.loads(json.dumps(base))
        
        # 1. New ID
        new_d["dialogue_id"] = f"dial_aug_{action_label[:3]}_{uuid.uuid4().hex[:6]}"
        
        # 2. Randomize Demographics (This causes ElevenLabs to use entirely different voices)
        new_d["domain"] = random.choice(DOMAINS)
        new_d["customer_profile"] = {
            "gender": random.choice(GENDERS),
            "age_group": random.choice(AGE_GROUPS),
            "accent": random.choice(CUSTOMER_ACCENTS)
        }
        new_d["engineer_profile"] = {
            "gender": random.choice(GENDERS),
            "accent": random.choice(ENGINEER_ACCENTS)
        }
        
        # 3. Random Train/Test Split (90/10)
        new_d["split"] = "train" if random.random() < 0.9 else "test"
        
        # 4. Perturb Prosody & Timing (Creates temporal and acoustic variance)
        for turn in new_d["turns"]:
            # Shift tts style weight by +/- 0.05
            shift = random.uniform(-0.05, 0.05)
            turn["tts_style_weight"] = max(0.0, min(1.0, turn["tts_style_weight"] + shift))
            turn["tts_style_weight"] = round(turn["tts_style_weight"], 2)
            
            # Shift offset by +/- 75ms
            time_shift = random.randint(-75, 75)
            turn["offset_ms"] = turn["offset_ms"] + time_shift
            
        new_dialogues.append(new_d)

# Shuffle and append
random.shuffle(new_dialogues)

import uuid # Import here in case missed above
with open(input_file, "a", encoding="utf-8") as f:
    for d in new_dialogues:
        f.write(json.dumps(d) + "\n")

print(f"Successfully appended {len(new_dialogues)} augmented dialogues. Total is now exactly 500.")
