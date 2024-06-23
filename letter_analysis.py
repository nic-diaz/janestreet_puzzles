import string
import random
import csv

state_population_dict = {}
state_list = []

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


def adjust_for_popluation(
    letters_count: dict[str, int],
    total_population: int,
    state_population_dict: dict[str, int],
):

    weights = {key: 0 for key in letters_count.keys()}
    for state, popluation in state_population_dict.items():
        population_weight = popluation / total_population

        state_letters = set()
        for letter in state:
            
            if letter in weights and letter not in state_letters:
                weights[letter] += population_weight * 100000
                state_letters.add(letter)

    for letter in letters_count:
        weights[letter] = weights[letter] + letters_count[letter]

    return weights


def letter_counts(states: list[str]):

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
    for letter, weight in weights.items():
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


letter_count = letter_counts(state_list)
weights = adjust_for_popluation(letter_count, total_population, state_population_dict)
normalized_weights = normalize_weights(weights)
grid = generate_weighted_grid(normalized_weights, state_population_dict)

breakpoint()
print(normalized_weights)

WEIGHTS = {
    "a": 0.2745777953719077,
    "b": 0.00016607996944490098,
    "c": 0.014451532421964449,
    "d": 0.006546849908647176,
    "e": 0.05744408569472926,
    "f": 0.001452030260116045,
    "g": 0.005177561852080407,
    "h": 0.01216277591193588,
    "i": 0.17731892350452413,
    "j": 0.00011041824801127295,
    "k": 0.005168743934726413,
    "l": 0.0256179625312402,
    "m": 0.009028566146397316,
    "n": 0.14742922728074995,
    "o": 0.1061213862402939,
    "p": 0.0009653567093642425,
    "r": 0.0463024912556483,
    "s": 0.07039779785859618,
    "t": 0.02279480697211818,
    "u": 0.003350333590099214,
    "v": 0.0016151754963079682,
    "w": 0.007138081679180274,
    "x": 0.000743247044592584,
    "y": 0.0038337602312709497,
    "z": 8.500988605322756e-05,
}


