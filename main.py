import sys

from puzzle import Puzzle
from plotter import plot_points, plot_two_points, analyze_distribution, plot_three_points
from letter_analysis import combine_weights, read_weights_from_file, normalize_weights


OLD_WEIGHTS = {
    "a": 0.11082419125285721,
    "b": 0.003238955854633758,
    "c": 0.043606550095286606,
    "d": 0.022700973353598945,
    "e": 0.05970183330297609,
    "f": 0.028294330008633344,
    "g": 0.020270205101292058,
    "h": 0.030982062705611442,
    "i": 0.09692577586296637,
    "j": 0.0043042964407431035,
    "k": 0.018069664296860015,
    "l": 0.06063992187111965,
    "m": 0.025151737035725985,
    "n": 0.09932962257398625,
    "o": 0.09233584953088123,
    "p": 0.008038905958553584,
    "r": 0.07721214086036975,
    "s": 0.06035350652983404,
    "t": 0.04108166105849405,
    "u": 0.016331678881959816,
    "v": 0.012595651475236986,
    "w": 0.025303750016762046,
    "x": 0.014484442653075592,
    "y": 0.024908106673486082,
    "z": 0.0033141866050560276
}


bias_1 = 0.2
bias_2 = 0.8
weights_1 = read_weights_from_file("weights_letter_count")
weights_2 = read_weights_from_file("weights_letters_adjusted_for_population")
combined_weights = combine_weights(weights_1, weights_2, bias_1, bias_2)
normalized_weights = normalize_weights(combined_weights)

puzzle_1 = Puzzle(weights_1)
points_weights_1 = puzzle_1.run_iterations(1000)
analyze_distribution(points_weights_1)

puzzle_2 = Puzzle(weights_2)
points_weights_2 = puzzle_2.run_iterations(1000)
analyze_distribution(points_weights_2)

puzzle_3 = Puzzle(normalized_weights)
points_weights_3 = puzzle_2.run_iterations(1000)
analyze_distribution(points_weights_3)

plot_three_points(points_weights_1, points_weights_2, points_weights_3)

# puzzle_1 = Puzzle(weights_1)
# #points_random = puzzle.run_iterations(1000, random=True)
# points_weights = puzzle_1.run_iterations(1000)
# plot_points(points_weights)
# #plot_two_points(points_random, points_weights)
# analyze_distribution(points_weights)

