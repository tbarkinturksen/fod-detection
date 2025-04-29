#################################################
# CONFIGURATION SETTINGS
#################################################
# File paths
INPUT_FILE = "input.txt"  # Input text file to process
OUTPUT_FILE = "output.txt"  # Output file for processed text

# Processing parameters
TRIM_CHARS = 49  # Number of characters to trim from the beginning of each line


def process_file():
    """
    Process a text file by trimming characters from the beginning of each line
    """
    print(f"Processing file: {INPUT_FILE}")
    
    # Read lines from input file
    with open(INPUT_FILE, "r") as infile:
        lines = infile.readlines()

    # Trim the specified number of characters from each line
    trimmed_lines = [line[TRIM_CHARS:].strip() for line in lines]

    # Write the result to output file
    with open(OUTPUT_FILE, "w") as outfile:
        for line in trimmed_lines:
            outfile.write(line + "\n")

    print(f"✅ Saved trimmed lines to: {OUTPUT_FILE}")
    print(f"   Processed {len(lines)} lines, removing first {TRIM_CHARS} characters from each")


if __name__ == "__main__":
    process_file()
