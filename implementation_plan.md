# Phase 1: Dialogue Script Generation Engine & Schema Specification

This document outlines the design and specifications for the initial phase of the Multimodal CSAT Speech Classification system. It focuses on the synthetic generation of the 500 dialogue scripts using an LLM to accurately model diverse regional accents, overlapping turn-taking, and granular emotional responses.

## User Review Required
> [!IMPORTANT]
> Please review this specification, particularly the JSON schema, duration heuristics, and the 4 generated sample dialogues. Once you approve this phase, we will proceed to Phase 2 (ElevenLabs Audio Synthesis & Dual-Channel Stitcher Pipeline).

## 1. Data Schema (`dialogue_schema.json`)
The generated scripts will be saved in JSONL format using the following schema. This structured data guarantees downstream compatibility with the audio synthesis and assembly pipeline.

```json
{
  "dialogue_id": "string (e.g., dial_001_urgent)",
  "target_duration_sec": "integer (Constraint: 25-35)",
  "action_label": "urgent_follow_up | at_risk_dissatisfied | standard_resolved | promoter_delighted",
  "fine_grained_emotion": "string (e.g., explosive anger, cold sarcasm, polite neutral, ecstatic praise)",
  "split": "train | test",
  "domain": "string (e-commerce | telecom | banking | tech support)",
  "customer_profile": {
    "gender": "male | female",
    "age_group": "young | middle-aged | elderly",
    "accent": "Singaporean English | Indian English | Chinese English | Malay English"
  },
  "engineer_profile": {
    "gender": "male | female",
    "accent": "string (e.g., Standard American, British, Neutral Asian)"
  },
  "turns": [
    {
      "speaker": "customer | engineer",
      "text": "string (Includes paralinguistic tags like [sigh], [scoff], aggressive capitalization)",
      "emotion_tag": "string (Maps to TTS prosody)",
      "tts_style_weight": "float (0.0 - 1.0)",
      "offset_ms": "integer (positive for inter-turn pause, negative for interruption/cut-in)"
    }
  ]
}
```

## 2. Word Count & Duration Heuristics
To consistently hit the **25–35 second** target duration in final audio, the LLM prompt enforces strict heuristics:
* **Speaking Rate Calculation:** The average conversational speaking rate is ~140 words per minute (WPM), or ~2.3 words per second.
* **Target Word Count:** To reach ~30 seconds, a dialogue requires **60 to 75 total words** across all turns.
* **Turn Constraints:** 
  * 4 to 8 conversational turns per dialogue.
  * Short turns: 3–8 words (acknowledgments, interruptions).
  * Long turns: 15–25 words (explanations, rants).
* **Offset Math:** 
  * Interruptions (`offset_ms`: -400 to -1200) effectively reduce overall audio duration. To compensate, word count should trend towards the higher end (70-75 words).
  * Polite pauses (`offset_ms`: +200 to +500) add to overall duration. Word count should trend lower (60-65 words).

## 3. Script Generation Prompt Template
We will use an LLM (e.g., Gemini 1.5 Pro) with strict JSON-mode structured output to generate the scripts in batch.

```text
You are an expert dialogue writer specializing in call-center linguistics, acoustic emotions, and regional Asian English accents.
Generate a multi-turn JSON dialogue script between a 'customer' and an 'engineer' based on the following parameters:

TARGET_CLASS: {action_label}
DOMAIN: {domain}
CUSTOMER_ACCENT: {accent}
TARGET_DURATION: 30 seconds (~65-75 words total)

Rules:
1. Adhere strictly to the JSON schema provided.
2. Ensure the text reflects the requested CUSTOMER_ACCENT (e.g., use subtle Singlish particles like 'lah', 'leh', 'meh' naturally for Singaporean English).
3. Inject paralinguistic cues in brackets (e.g., [sigh], [angry gasp], [scoff], [warm laugh]) and use ALL CAPS for shouting to guide the TTS engine.
4. Manage turn-taking dynamically via the 'offset_ms' field:
   - For 'urgent_follow_up' and 'at_risk_dissatisfied', the customer MUST interrupt the engineer at least once using a negative offset (e.g., -600 to -1000).
   - For 'standard_resolved' and 'promoter_delighted', use polite positive offsets (200 to 500).
5. The 'tts_style_weight' should map to the emotion: 0.7-0.9 for high intensity (rage/delight), 0.1-0.2 for neutral/calm.
```

## 4. Full Dialogue Examples (4 Classes)

### Example 1: `urgent_follow_up`
**Domain:** Telecom | **Accent:** Singaporean English | **Emotion:** Explosive Anger
```json
{
  "dialogue_id": "dial_001_urgent",
  "target_duration_sec": 29,
  "action_label": "urgent_follow_up",
  "fine_grained_emotion": "explosive anger and severe impatience",
  "split": "train",
  "domain": "telecom",
  "customer_profile": {
    "gender": "male",
    "age_group": "middle-aged",
    "accent": "Singaporean English"
  },
  "engineer_profile": {
    "gender": "female",
    "accent": "Neutral Asian"
  },
  "turns": [
    {
      "speaker": "engineer",
      "text": "Thank you for calling StarHub support. My name is Amanda. May I have your account number please?",
      "emotion_tag": "polite_neutral",
      "tts_style_weight": 0.2,
      "offset_ms": 200
    },
    {
      "speaker": "customer",
      "text": "[angry gasp] Don't ask me for account number again! I called three times already today! Your internet is down AGAIN, completely useless lah!",
      "emotion_tag": "explosive_anger",
      "tts_style_weight": 0.85,
      "offset_ms": -400
    },
    {
      "speaker": "engineer",
      "text": "I deeply apologize for the repeated downtime, sir. I see there's an outage in your area and we are currently working to—",
      "emotion_tag": "apologetic_calm",
      "tts_style_weight": 0.3,
      "offset_ms": 100
    },
    {
      "speaker": "customer",
      "text": "CANCEL IT! Just cancel the whole line! I am switching to Singtel tomorrow. Absolute rubbish service!",
      "emotion_tag": "shouting_furious",
      "tts_style_weight": 0.9,
      "offset_ms": -800
    }
  ]
}
```

### Example 2: `at_risk_dissatisfied`
**Domain:** Banking | **Accent:** Malay English | **Emotion:** Cold Sarcasm & Exasperation
```json
{
  "dialogue_id": "dial_002_atrisk",
  "target_duration_sec": 32,
  "action_label": "at_risk_dissatisfied",
  "fine_grained_emotion": "cold sarcasm and exasperation",
  "split": "train",
  "domain": "banking",
  "customer_profile": {
    "gender": "female",
    "age_group": "young",
    "accent": "Malay English"
  },
  "engineer_profile": {
    "gender": "male",
    "accent": "Standard American"
  },
  "turns": [
    {
      "speaker": "customer",
      "text": "[heavy sigh] So you're telling me, my card is blocked while I'm stuck overseas, and I have to visit a branch... in Kuala Lumpur?",
      "emotion_tag": "exasperated",
      "tts_style_weight": 0.6,
      "offset_ms": 300
    },
    {
      "speaker": "engineer",
      "text": "Unfortunately, for security protocol regarding suspected fraud, physical verification is the only way we can unlock the card, ma'am.",
      "emotion_tag": "matter_of_fact",
      "tts_style_weight": 0.2,
      "offset_ms": 200
    },
    {
      "speaker": "customer",
      "text": "[scoff] Wow. Fantastic. Really a world-class digital bank. So helpful.",
      "emotion_tag": "cold_sarcasm",
      "tts_style_weight": 0.65,
      "offset_ms": -300
    },
    {
      "speaker": "engineer",
      "text": "I understand this is frustrating. I can offer you an emergency cash advance via wire transfer if—",
      "emotion_tag": "sympathetic",
      "tts_style_weight": 0.3,
      "offset_ms": 200
    },
    {
      "speaker": "customer",
      "text": "Whatever. Just leave it blocked. I'll figure it out myself.",
      "emotion_tag": "flat_hopeless",
      "tts_style_weight": 0.1,
      "offset_ms": -500
    }
  ]
}
```

### Example 3: `standard_resolved`
**Domain:** Tech Support | **Accent:** Chinese English | **Emotion:** Polite Neutral
```json
{
  "dialogue_id": "dial_003_resolved",
  "target_duration_sec": 28,
  "action_label": "standard_resolved",
  "fine_grained_emotion": "matter-of-fact and polite",
  "split": "train",
  "domain": "tech support",
  "customer_profile": {
    "gender": "male",
    "age_group": "elderly",
    "accent": "Chinese English"
  },
  "engineer_profile": {
    "gender": "male",
    "accent": "Neutral Asian"
  },
  "turns": [
    {
      "speaker": "customer",
      "text": "Hello, my laptop is asking for a BitLocker recovery key. I am not sure where to find this.",
      "emotion_tag": "confused_neutral",
      "tts_style_weight": 0.15,
      "offset_ms": 300
    },
    {
      "speaker": "engineer",
      "text": "Not a problem at all. If you log into your Microsoft account on your phone, you will find it under the Devices section.",
      "emotion_tag": "helpful_polite",
      "tts_style_weight": 0.2,
      "offset_ms": 250
    },
    {
      "speaker": "customer",
      "text": "Ah, okay, let me check now. [pause] Yes, I see a long string of numbers here.",
      "emotion_tag": "relieved_neutral",
      "tts_style_weight": 0.2,
      "offset_ms": 400
    },
    {
      "speaker": "engineer",
      "text": "Perfect. Just type those numbers exactly as they appear into your laptop, and it will unlock.",
      "emotion_tag": "encouraging",
      "tts_style_weight": 0.25,
      "offset_ms": 200
    },
    {
      "speaker": "customer",
      "text": "It is booting up now. Thank you for the quick help.",
      "emotion_tag": "polite_acknowledgment",
      "tts_style_weight": 0.15,
      "offset_ms": 350
    }
  ]
}
```

### Example 4: `promoter_delighted`
**Domain:** E-commerce | **Accent:** Indian English | **Emotion:** Ecstatic Praise
```json
{
  "dialogue_id": "dial_004_promoter",
  "target_duration_sec": 31,
  "action_label": "promoter_delighted",
  "fine_grained_emotion": "ecstatic praise and genuine gratitude",
  "split": "train",
  "domain": "e-commerce",
  "customer_profile": {
    "gender": "female",
    "age_group": "middle-aged",
    "accent": "Indian English"
  },
  "engineer_profile": {
    "gender": "female",
    "accent": "British"
  },
  "turns": [
    {
      "speaker": "customer",
      "text": "Hi! I just realized I entered the completely wrong shipping address for my daughter's wedding gift. Is it too late to change it?",
      "emotion_tag": "anxious_energetic",
      "tts_style_weight": 0.5,
      "offset_ms": 200
    },
    {
      "speaker": "engineer",
      "text": "Let me check the dispatch status for you. Oh, it was just about to leave the warehouse. I've intercepted it and updated the address.",
      "emotion_tag": "cheerful_reassuring",
      "tts_style_weight": 0.4,
      "offset_ms": 300
    },
    {
      "speaker": "customer",
      "text": "[warm laugh] Oh my goodness, are you serious? You are an absolute lifesaver! I was so stressed out about this!",
      "emotion_tag": "ecstatic_praise",
      "tts_style_weight": 0.8,
      "offset_ms": -100
    },
    {
      "speaker": "engineer",
      "text": "You're very welcome! It will arrive at the new address by Thursday.",
      "emotion_tag": "happy_polite",
      "tts_style_weight": 0.5,
      "offset_ms": 250
    },
    {
      "speaker": "customer",
      "text": "Thank you, thank you so much! Brilliant service, truly. Have a wonderful day!",
      "emotion_tag": "effusive_gratitude",
      "tts_style_weight": 0.75,
      "offset_ms": 200
    }
  ]
}
```

## Next Steps

1. Create a `generate_scripts.py` orchestrator script using `google-genai` or `openai` python package to asynchronously prompt the LLM, iterate over the domain/accent/class matrix, and output exactly 500 valid entries to `data/raw/dialogues.jsonl`.
2. Validate the corpus for duration boundaries using the word count approximations.
3. Determine if updates to `.agents` context files (like `handoff.md` or `walkthrough.md`) are needed after generation.

**Are you ready to approve this schema and generation design?**
