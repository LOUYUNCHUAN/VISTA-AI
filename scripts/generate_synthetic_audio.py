import os
import argparse
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import save

# Load environment variables
load_dotenv()

def generate_bullying_audio(text, output_path):
    """
    Generates synthetic emotional audio using the ElevenLabs API.
    """
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("❌ Error: ELEVENLABS_API_KEY not found in .env file.")
        print("Please get an API key at https://elevenlabs.io and add it to your .env file like this:")
        print("ELEVENLABS_API_KEY=your_actual_api_key_here")
        return

    print(f"Generating audio for text: '{text}'...")
    
    # Initialize the client
    client = ElevenLabs(api_key=api_key)
    
    try:
        # Generate the audio
        audio = client.generate(
            text=text,
            voice="Rachel", # Default voice
            model="eleven_multilingual_v2"
        )
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the audio
        save(audio, output_path)
        print(f"✅ Successfully generated and saved to {output_path}")
        
    except Exception as e:
        print(f"❌ Error calling ElevenLabs API: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic bullying audio.")
    
    # Default aggressive text to test the emotional prosody
    default_text = "Hey! What are you looking at? You are absolutely worthless. Give me your money right now before I hurt you!"
    
    parser.add_argument("--text", type=str, default=default_text, help="The text to synthesize.")
    parser.add_argument("--output", type=str, default="data/raw/synthetic_bullying_1.wav", help="Output file path.")
    
    args = parser.parse_args()
    generate_bullying_audio(args.text, args.output)
