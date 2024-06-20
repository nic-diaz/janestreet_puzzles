import csv
import sys
import random
import string

import numpy as np
import matplotlib.pyplot as plt
MAX_INT = sys.maxsize
MIN_INT = -sys.maxsize - 1
import time



class Puzzle:
    
    def _get_min_max_state_len(self, state_list: list[str]) -> tuple[int, int]:
        """ Returns min and max number of letters in a state"""
        min_len = MAX_INT
        max_len = MIN_INT
        
        for state in state_list:
            
            if len(state) > max_len:
                max_len = len(state)
                
            if len(state) < min_len:
                min_len = len(state)
                
        return min_len, max_len
        
    
    def __init__(self, csv_file: str, size=5) -> None:
        
        # Initialize dictionary and list
        state_population_dict = {}
        state_list = []

        # Read CSV data from file
        with open(csv_file, mode='r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip header row
            for row in csv_reader:
                state, population = row
                state_population_dict[state.strip()] = int(population.strip())
                state_list.append(state.strip())
                
        min_len, max_len = self._get_min_max_state_len(state_list)

        self.state_population_dict: dict[str, int] = state_population_dict
        self.state_list: list[str] = state_list
        self.min_state_len: int = min_len
        self.max_state_len: int = max_len
        
        self.size = size
        #self.board: list[str] = self.generate_random_board(size)
        
        self.board: list[str] = ["ohiox", "xxxxx", "xxxxx", "xxxxx", "xxxxx"]
        self.rows = size
        self.columns = size
        
        self.directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),  # Up, Down, Left, Right
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # Diagonals: Up-Left, Up-Right, Down-Left, Down-Right
        ]
        
    def generate_random_board(self, size: int = None) -> list[str]:
        
        if size is None:
            size = self.size
            
        """ Returns a board with random letters of [size X size]"""
        random_board = []
        for _ in range(size):
            random_board.append(''.join(random.choices(string.ascii_lowercase, k=size)))
            
        return random_board
            
    
    
    def _different_by_one_letter(self, word: str, state: str) -> bool:
        """ Returns bool whether word differs to state by exactly one letter"""
        if len(word) != len(state):
            return False
        
        differences = 0
        for idx, letter in enumerate(word):
            if state[idx] != letter:
                differences +=1 
            
            if differences > 1:
                return False
            
        return differences == 1
            
        
    
    def count_as_state(self, word: str, possible_states: set) -> bool:
        """ Returns bool whether word counts as a state"""
        
        if len(word) < self.min_state_len or len(word) > self.max_state_len:
            return False
        
        for state in possible_states:
            
            # Perfect match
            if word == state:
                return True
            
            # Different by one letter
            if self._different_by_one_letter(word, state):
                return True
        
        return False
            
            
            
    def calculate_score(self, states: set) -> int:
        points = 0
        for state in states:
            assert state in self.state_population_dict, f"Error, {state} not in population dictionary"
            points += self.state_population_dict[state]
            
        if points > 165379868:
            print(f"Score qualifies for leaderboard! Saving board...")
            filename = f"leadership_board_{points}.txt"
            self.save_board_to_file(filename=filename)
            
            
        #print(f"Points: {points}")
        return points
    
    def save_board_to_file(self, filename: str):
        with open(filename, 'w') as file:
            for string in self.board:
                file.write(string + '\n')
    
    def _update_possible_states(self, word: str, possible_states: list[str]) -> list[str]:
        """ Returns list of possible a word could turn into """
        
        
        if len(word) == 1:
            return possible_states
        
        possible_states_copy = possible_states.copy()
        for state in possible_states_copy:
            if len(word) > len(state):
                possible_states.remove(state)
                continue
            
            truncated_state = state[:len(word)]
            
            if truncated_state == word:
                #print(f"{word} could become {state}")
                continue # Keep state in possible_states
            elif self._different_by_one_letter(truncated_state, word):
                #print(f"{word} could become {state}")
                continue # Keep state in possible_states
            else:
                #print(f"Removing {state}")
                possible_states.remove(state)
                
        return possible_states
    
    def run_forever(self, threshold: int = 80000000) -> None:
        
        while(True):
            self.board = self.generate_random_board()
            matched_states = self.search_states()
            points = self.calculate_score(matched_states)
            
            if points> threshold:
                self.save_board_to_file(f"board_{points}.txt")        

    def run_iterations(self, iterations: int  = 1000000, threshold: int = 80000000) -> None:
        
        max_points = 0
        all_points = []
                
        start_time = time.time()
        for _ in range(iterations):
            self.board = self.generate_random_board()
            matched_states = self.search_states()
            points = self.calculate_score(matched_states)
            
            
            if points> threshold:
                all_points.append(points)
                self.save_board_to_file(f"board_{points}.txt")
                
            max_points = max(max_points, points)
        
        all_points.append(0) # Min Value
        all_points.append(165379868) # Target Value
        
        # Normalize Points
        all_points = np.array(all_points)
        min_val = np.min(all_points)
        max_val = np.max(all_points)
        normalized_points = (all_points - min_val) / (max_val - min_val)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"Elapsed time: {elapsed_time}")
        print(f"Max score found: {max_points}")
        
        # Plot the normalized points
        plt.figure(figsize=(10, 6))
        plt.plot(normalized_points, 'o', markersize=1)
        plt.title('Normalized Points')
        plt.xlabel('Index')
        plt.ylabel('Normalized Value')
        plt.show()
        
        
    def find_states_in_board(self, filename: str) -> None:
        with open(filename, mode='r') as file:
            board = file.readlines()
            board = [line.strip() for line in board]
        self.board = board
        print(self.search_states())
        
        return 
                
        
    def search_states(self) -> set:
        matched_states = set()
        
        for r in range(self.rows):
            for c in range(self.columns):
                possible_states = self.state_list
                self.dfs(r, c, "", possible_states, matched_states)
        return matched_states
    
    def convert_path_to_state(self, current_path: str, possible_states: list[str]) -> str:
        for state in possible_states:
            if current_path == state:
                return state
            
            if self._different_by_one_letter(current_path, state):
                return state
        return ""

    def dfs(self, r, c, current_path, possible_states: list[str], matched_states: set):
        if (r < 0 or r >= self.rows or c < 0 or c >= self.columns or len(current_path) > self.max_state_len or len(possible_states) == 0):
            return

        current_path += self.board[r][c]
        #print(f"Current path: {current_path}")
        
        possible_states = possible_states.copy()
        new_possible_states = self._update_possible_states(current_path, possible_states)
        
        if len(new_possible_states) == 0:
            return
        
        if self.count_as_state(current_path, possible_states):
            state = self.convert_path_to_state(current_path, possible_states)
            matched_states.add(state)
        


        for dr, dc in self.directions:
            new_r, new_c = r + dr, c + dc
            self.dfs(new_r, new_c, current_path, new_possible_states, matched_states)



puzzle = Puzzle("states.csv")
#puzzle.run_iterations()
puzzle.run_forever()
#puzzle.find_states_in_board("board_86394544.txt")

# puzzle.run()
# states = ["texas", "california", "ohio", "new york", "michigan", "florida", "georgia", "virginia", "washington", "pennsylvania"]
# puzzle.calculate_score(states)

