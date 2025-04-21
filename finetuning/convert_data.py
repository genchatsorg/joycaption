import json
import requests
import os
from urllib.parse import urlparse
from pathlib import Path

def download_image(url, save_path):
    """Download image from URL and save to specified path"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False

def get_filename_from_url(url):
    """Extract filename from URL"""
    parsed = urlparse(url)
    return os.path.basename(parsed.path)

def main():
    # Create output directories
    output_dir = Path("data/downloaded_images")
    output_dir.mkdir(exist_ok=True)
    
    # Load the JSON data
    with open('data/infl_links.json', 'r') as f:
        data = json.load(f)
    
    # List to store the converted format
    converted_data = []
    
    # Process each entry
    for entry in data:
        if entry['type'] == 'IMAGE':
            url = entry['url']
            tags = json.loads(entry['tags'])  # Parse tags JSON string
            filename = get_filename_from_url(url)
            save_path = output_dir / filename
            
            # Create entry in the new format
            converted_entry = {
                "messages": [
                    {
                        "role": "user",
                        "content": "Write a list of tags for this image."
                    },
                    {
                        "role": "assistant",
                        "content": ",".join(tags)
                    }
                ],
                "images": [
                    f"data/downloaded_images/{filename}"
                ]
            }
            
            converted_data.append(converted_entry)
            
            # Download image
            print(f"Downloading {filename}...")
            if download_image(url, save_path):
                print(f"Successfully downloaded {filename}")
            else:
                print(f"Failed to download {filename}")
    
    # Save converted data to a JSON file
    output_file = output_dir / 'converted_data.json'
    with open(output_file, 'w') as f:
        json.dump(converted_data, f, indent=2)
    
    print(f"\nDownloaded images and saved converted data to {output_file}")

if __name__ == "__main__":
    main()