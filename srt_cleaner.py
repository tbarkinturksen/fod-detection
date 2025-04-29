import re
import os
from datetime import datetime, timedelta

#################################################
# CONFIGURATION SETTINGS
#################################################
# File paths
INPUT_PATH = "srt/DJI_0252.SRT"  # Path to input SRT file
OUTPUT_PATH = "srt/export/DJI_0252_cleaned.srt"  # Path to output cleaned SRT file

# Processing parameters
INTERVAL_SEC = 0.5  # Time interval between subtitle entries in seconds


def parse_dji_srt(filename):
    """
    Parse a DJI SRT subtitle file
    
    Args:
        filename: Path to the SRT file
        
    Returns:
        List of subtitle entries
    """
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    entries = []
    current_entry = []
    for line in lines:
        if line.strip().isdigit():
            if current_entry:
                entries.append(current_entry)
            current_entry = [line.strip()]
        elif line.strip():
            current_entry.append(line.strip())
    if current_entry:
        entries.append(current_entry)

    print(f"Parsed {len(entries)} entries.")
    return entries


def extract_time(entry):
    """
    Extract timestamp from subtitle entry
    
    Args:
        entry: Subtitle entry lines
        
    Returns:
        Datetime object or None if no timestamp found
    """
    for line in entry:
        if "-->" in line:
            start = line.split(" --> ")[0]
            return datetime.strptime(start, "%H:%M:%S,%f")
    return None


def extract_gps(entry):
    """
    Extract GPS coordinates from subtitle entry
    
    Args:
        entry: Subtitle entry lines
        
    Returns:
        Formatted GPS string or None if no coordinates found
    """
    gps_line = next((line for line in entry if "[latitude:" in line and "[longitude:" in line), "")
    lat_match = re.search(r"\[latitude:\s*([0-9\.\-]+)\]", gps_line)
    lon_match = re.search(r"\[longitude:\s*([0-9\.\-]+)\]", gps_line)

    if lat_match and lon_match:
        lat = lat_match.group(1)
        lon = lon_match.group(1)
        return f"Lat: {lat}, Lng: {lon}"
    return None


def format_time(dt):
    """
    Format datetime object to SRT timestamp format
    
    Args:
        dt: Datetime object
        
    Returns:
        Formatted timestamp string
    """
    return dt.strftime("%H:%M:%S,%f")[:-3]  # Remove last 3 digits of microseconds


def downsample(entries, interval_sec=0.5):
    """
    Downsample subtitle entries to specified time interval
    
    Args:
        entries: List of subtitle entries
        interval_sec: Time interval between entries in seconds
        
    Returns:
        List of processed subtitle entries in SRT format
    """
    selected_entries = []
    last_written_time = None

    # First pass: select entries based on interval
    for entry in entries:
        timestamp = extract_time(entry)
        if not timestamp:
            continue

        if last_written_time is None or (timestamp - last_written_time).total_seconds() >= interval_sec:
            gps_text = extract_gps(entry)
            if gps_text:
                selected_entries.append((timestamp, gps_text))
                last_written_time = timestamp

    # Second pass: create continuous subtitles
    output = []
    for i, (timestamp, gps_text) in enumerate(selected_entries):
        count = i + 1
        start_time = format_time(timestamp)

        # Set end time to the start time of the next entry, or add 10 seconds for the last entry
        if i < len(selected_entries) - 1:
            end_time = format_time(selected_entries[i + 1][0])
        else:
            end_time = format_time(timestamp + timedelta(seconds=10))

        time_range = f"{start_time} --> {end_time}"
        output.append(f"{count}\n{time_range}\n{gps_text}")

    print(f"✅ Kept {len(output)} entries after downsampling.")
    return output


def save_output(entries, output_file):
    """
    Save processed subtitle entries to output file
    
    Args:
        entries: List of formatted subtitle entries
        output_file: Path to output file
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(entries))
    
    print(f"✅ Saved cleaned SRT to: {output_file}")


def main():
    """
    Main function to run the SRT cleaning process
    """
    print(f"Processing SRT file: {INPUT_PATH}")
    entries = parse_dji_srt(INPUT_PATH)
    filtered = downsample(entries, interval_sec=INTERVAL_SEC)
    save_output(filtered, OUTPUT_PATH)


if __name__ == "__main__":
    main()
