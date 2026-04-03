#!/usr/bin/env python3
import os
import time

def remove_commands_from_trace(trace_file, commands):
    """
    Remove lines containing specified commands from a trace file
    
    Args:
        trace_file (str): Path to the trace file
        commands (list): List of commands to remove
    """
    print(f"Current working directory: {os.getcwd()}")
    print(f"Checking for file: {trace_file}")
    print(f"File exists: {os.path.exists(trace_file)}")
    
    if not os.path.exists(trace_file):
        print(f"Error: File {trace_file} does not exist.")
        return False
    
    print(f"File size: {os.path.getsize(trace_file)} bytes")
    
    # Read the file
    with open(trace_file, 'r') as f:
        lines = f.readlines()
    
    print(f"Read {len(lines)} lines")
    
    # Filter out lines containing any of the commands
    filtered_lines = []
    for line in lines:
        if not any(cmd in line for cmd in commands):
            filtered_lines.append(line)
    
    print(f"Filtered to {len(filtered_lines)} lines")
    
    # Write back the filtered content
    with open(trace_file, 'w') as f:
        f.writelines(filtered_lines)
    
    print(f"Successfully removed lines containing commands: {', '.join(commands)}")
    print(f"Filtered {len(lines)} lines down to {len(filtered_lines)} lines")
    print(f"File now exists: {os.path.exists(trace_file)}")
    if os.path.exists(trace_file):
        print(f"File size after: {os.path.getsize(trace_file)} bytes")
    return True

if __name__ == "__main__":
    trace_file = "/home/vel/lerobot/syscall-analyze/full_trace.txt"
    commands_to_remove = ["speech", "vncagent", "ANGLE"]
    print(f"Starting at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    result = remove_commands_from_trace(trace_file, commands_to_remove)
    print(f"Result: {result}")
    print(f"Ending at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Final file existence: {os.path.exists(trace_file)}")
