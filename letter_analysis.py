import string
import random
import csv
import json

state_population_dict = {}
state_list = []
letters_set = set()

total_population = 0
# Read CSV data from file
with open("states.csv", mode="r") as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip header row
    for row in csv_reader:
        state, population = row
        state = state.replace(" ", "")
        state_population_dict[state.strip()] = int(population.strip())
        state_list.append(state.strip())
        total_population += int(population.strip())
        
        for letter in state:
            letters_set.add(letter)


def letters_adjusted_for_popluation(
    letters_set: set,
    total_population: int,
    state_population_dict: dict[str, int],
):

    weights = {letter: 0 for letter in letters_set}
    for state, popluation in state_population_dict.items():
        population_weight = popluation / total_population

        state_letters = set()
        for letter in state:
            
            if letter in weights and letter not in state_letters:
                weights[letter] += population_weight
                state_letters.add(letter)



    return weights


def letter_counts(states: list[str]):
    """ Returns dictionary with
            key: letters
            value: number of occurrences
            
    """

    # Initialize a dictionary with keys a-z and values 0
    letter_counts = {letter: 0 for letter in string.ascii_lowercase}

    # Count the total number of letters
    total_letters = 0

    # Iterate through each word and count the letters
    for state in states:
        for (
            char
        ) in state.lower():  # Convert to lowercase to count all letters uniformly
            if char in letter_counts:
                letter_counts[char] += 1
                total_letters += 1

    copy = letter_counts.copy()

    for letter, count in copy.items():
        if count == 0:
            del letter_counts[letter]

    return letter_counts


def normalize_weights(weights: dict[str, int]) -> dict[str, int]:

    total_weight = 0
    for _, weight in weights.items():
        total_weight += weight

    return {letter: weight / total_weight for letter, weight in weights.items()}


def generate_weighted_grid(
    frequencies: dict[str, float], state_population_dict: dict[str, int]
):
    grid_size = 5

    # Create a list of letters and their corresponding weights
    letters = list(frequencies.keys())
    weights = list(frequencies.values())

    # Todo: Adjust weights based on population

    # Generate the grid
    grid = []
    for _ in range(grid_size):
        row = "".join(random.choices(letters, weights, k=grid_size))
        grid.append(row)

    return grid

def read_weights_from_file(filename: str):
    with open(f"{filename}.json", 'r') as json_file:
        return json.load(json_file)

def write_weights_to_file(filename: str) -> None:
    with open(f"{filename}.json", 'w') as json_file:
        json.dump(normalized_weights, json_file, indent=4)

def combine_weights(weights_1, weights_2, bias_1, bias_2):
    """
    Combines two sets of weights with given biases.
    
    Parameters:
    - weights_1: dict, first set of weights
    - weights_2: dict, second set of weights
    - bias_1: float, bias for the first set of weights (e.g., 0.4 for 40%)
    - bias_2: float, bias for the second set of weights (e.g., 0.6 for 60%)
    
    Returns:
    - dict, combined weights
    """
    weights_combined = {}
    
    for key in weights_1:
        if key in weights_2:
            weights_combined[key] = bias_1 * weights_1[key] + bias_2 * weights_2[key]
        else:
            weights_combined[key] = bias_1 * weights_1[key]
    
    for key in weights_2:
        if key not in weights_combined:
            weights_combined[key] = bias_2 * weights_2[key]
    
    return weights_combined


# weights_1 = letter_counts(state_list)
# normalized_weights = normalize_weights(weights_1)
# write_weights_to_file("weights_letter_count")


# weights_2 = letters_adjusted_for_popluation(letters_set, total_population, state_population_dict)
# normalized_weights = normalize_weights(weights_2)
# write_weights_to_file("weights_letters_adjusted_for_population")



# bias_1  = 0.4
# bias_2 = 0.6
# combined_weights = combine_weights(weights_1, weights_2, bias_1, bias_2)
# normalize_weights = normalize_weights(combined_weights)
# write_weights_to_file("combined_40_count_60_pop")