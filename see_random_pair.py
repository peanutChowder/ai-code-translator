import json
import random


def print_random_pairs(file_path, num_pairs=5):
    """
    Prints a specified number of random Java-Python code pairs from a JSON file.

    Args:
    - file_path (str): Path to the JSON file containing the preprocessed data.
    - num_pairs (int): Number of random pairs to display.
    """
    try:
        # Load the preprocessed data
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Ensure there are enough pairs to select
        if len(data) < num_pairs:
            print(f"File contains fewer than {num_pairs} pairs. Displaying all available pairs instead.")
            num_pairs = len(data)

        # Select random pairs
        random_pairs = random.sample(data, num_pairs)

        # Print the selected pairs
        for i, pair in enumerate(random_pairs, start=1):
            print(f"\nPair {i}:")
            print("Java Code:")
            print(pair.get("java_processed", "No Java code found"))
            print("\nPython Code:")
            print(pair.get("python_processed", "No Python code found"))
            print("-" * 80)

    except FileNotFoundError:
        print(f"Error: File not found at path '{file_path}'. Please check the file path.")
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON. Please ensure the file is properly formatted.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# Example usage
file_path = "processed_pairs.json"
print_random_pairs(file_path)
