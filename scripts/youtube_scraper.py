import os
import argparse
import yt_dlp

def download_youtube_audio(url, output_dir="data/raw"):
    """
    Downloads the audio from a YouTube video and converts it to WAV format
    for the VISTA-AI project.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure yt-dlp to extract audio only and convert to wav
    ydl_opts = {
        'format': 'bestaudio/best',
        # Save file with the video title as the name
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': False,
    }
    
    print(f"Downloading audio from: {url}")
    print(f"Target directory: {output_dir}")
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"\n✅ Successfully downloaded and saved to '{output_dir}/'")
    except Exception as e:
        print(f"\n❌ Error downloading video: {e}")
        print("\nNote: If you received an FFmpeg error, yt-dlp requires FFmpeg to convert the file to WAV.")
        print("You can easily install it on Mac by running: brew install ffmpeg")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download YouTube video audio for the VISTA-AI dataset.")
    parser.add_argument("url", help="The YouTube URL to download")
    parser.add_argument("--output", default="data/raw", help="Output directory (default: data/raw)")
    
    args = parser.parse_args()
    download_youtube_audio(args.url, args.output)
