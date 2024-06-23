import csv
import sys
import random
import string

import numpy as np
from numpy.typing import NDArray


import matplotlib.pyplot as plt

MAX_INT = sys.maxsize
MIN_INT = -sys.maxsize - 1
TARGET_SCORE = 165379868

class Puzzle:
    csv_file: str = "states.csv"
    size: int = 5
    
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
        
    
    def __init__(self, weights: dict[str, float]) -> None:
        

        # Get state info from CSV
        state_population_dict = {}
        state_list = []
        with open(self.csv_file, mode='r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip header row
            for row in csv_reader:
                state, population = row
                state = state.replace(" ", "")
                state_population_dict[state.strip()] = int(population.strip())
                state_list.append(state.strip())
                
        min_len, max_len = self._get_min_max_state_len(state_list)

        self.state_population_dict: dict[str, int] = state_population_dict
        self.state_list: list[str] = state_list
        self.min_state_len: int = min_len
        self.max_state_len: int = max_len
        self.weights = weights
        
        
        self.board: list[str] = []
        self.rows = self.size
        self.columns = self.size
        
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
            
        if points >= TARGET_SCORE:
            print(f"Score qualifies for leaderboard! Saving board...")
            filename = f"leadership_board_{points}.txt"
            self.save_board_to_file(filename=filename)
                        
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
    
    def run_forever(self) -> None:
                
        while(True):
            self.board = self.generate_weighted_random_grid()
            matched_states = self.search_states()
            points = self.calculate_score(matched_states)
            
            if points> (TARGET_SCORE*0.8):
                self.save_board_to_file(f"board_{points}_weighted.txt")   

    def run_iterations(self, iterations: int  = 100000, random=False) -> NDArray[np.float64]:
        
        max_points = 0
        all_points= []
                
        for _ in range(iterations):
            
            if random:
                self.board = self.generate_random_board()
            else:
                self.board = self.generate_weighted_random_grid()
            
            matched_states = self.search_states()
            points = self.calculate_score(matched_states)
            all_points.append(points)
            
            if points> (TARGET_SCORE*0.8):
                self.save_board_to_file(f"board_{points}_weighted.txt")
                
            max_points = max(max_points, points)
        

        
        all_points = np.array(all_points)
        min_val = np.min(all_points)
        max_val = np.max(all_points)
        median_val = int(np.median(all_points))
        average_val = int(np.mean(all_points))
        
        print(f"\n\n*****RESULTS FOR {iterations} ITERATIONS*****\n")
        print(f"     - MIN points: {min_val:,}")
        print(f"     - MAX points: {max_val:,}")
        print(f"     - MEDIAN score: {median_val:,}")
        print(f"     - AVERAGE score: {average_val:,}\n")
        
        return all_points / TARGET_SCORE
        
        
    def print_states_in_board(self, filename: str) -> None:
        with open(filename, mode='r') as file:
            board = file.readlines()
            board = [line.strip() for line in board]
        self.board = board
        found_states = self.search_states(board)
        print(f"Board: {board}")
        print(f"States: {found_states}")
                
        
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

    def generate_weighted_random_grid(self):
        grid_size = self.size
        
        # Create a list of letters and their corresponding weights
        letters = list(self.weights.keys())
        weights = list(self.weights.values())
        
        # Generate the grid
        grid = []
        for _ in range(grid_size):
            row = ''.join(random.choices(letters, weights, k=grid_size))
            grid.append(row)
        
        return grid
    