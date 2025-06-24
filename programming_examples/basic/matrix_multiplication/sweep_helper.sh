#!/bin/bash

# --- Configuration ---

# Regex to match an integer (positive or negative)
# This will match the *first* such integer on the line.
# If your integers are always positive, you can simplify to '[0-9]+'
INTEGER_REGEX='[+-]?[0-9]+'

EXPECTED_ARGS=5
if [[ "$#" -ne "$EXPECTED_ARGS" ]]; then
    echo "Usage: $0 <file_path> <line1> <r> <s> <t>"
    echo "Replaces integers in three consecutive lines of the given file."
    echo "Error: Incorrect number of arguments. Expected $EXPECTED_ARGS, got $#."
    exit 1
fi

FILE_PATH="$1"
LINE_NUMS=("$2" "$(($2 + 1))" "$(($2 + 2))")
NEW_VALUES=("$3" "$4" "$5")

# --- Helper Function to Process a File ---
# Arguments:
# $1: file_path
# $2: array of line numbers (passed as string, e.g., "10 11 12")
# $3: array of new values (passed as string, e.g., "999 888 777")
process_file() {
    local file_path="$1"
    # Convert string arguments back to arrays
    local -a line_nums=($2)
    local -a new_values=($3)

    if [[ ! -f "$file_path" ]]; then
        echo "Error: File '$file_path' not found."
        return 1
    fi

    # echo "Modifying file: $file_path"

    # Construct the sed commands
    local sed_commands=""
    for i in "${!line_nums[@]}"; do
        local line_num="${line_nums[$i]}"
        local new_val="${new_values[$i]}"
        # Add a sed command for each line:
        # <line_num>s/<regex_to_match_integer>/<new_value>/
        # The 's' command by default replaces the first match on the line.
        sed_commands+="${line_num}s/${INTEGER_REGEX}/${new_val}/; "
    done

    # Execute sed command
    # -i modifies the file in-place.
    # Using a temporary variable for sed_commands to avoid issues with quoting if done directly.
    if sed -i -E "$sed_commands" "$file_path"; then
        echo "Successfully modified '$file_path'."
    else
        echo "Error modifying '$file_path' with sed."
        return 1
    fi
}

# --- Main Script ---

# Process File 1
# Pass arrays as space-separated strings
process_file "$FILE_PATH" "${LINE_NUMS[*]}" "${NEW_VALUES[*]}"

# # Process File 2
# process_file "$FILE2_PATH" "${FILE2_LINE_NUMS[*]}" "${NEW_VALUES[*]}"
