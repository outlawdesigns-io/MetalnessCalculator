# __main__.py
import pandas as pd
from os import path
from MetalnessCalculator.MetalnessCalculator import MetalnessCalculator

def main():
    """
    Train MetalnessCalculator to determine the 'metalness' of input
    Train and interact with...
    """
    metal_data_path = input("Path to the metal dataset: ")
    control_data_path = input("Path to the control dataset: ")
    metal_data = pd.read_csv(metal_data_path, sep=',', escapechar='\\')
    control_data = pd.read_csv(control_data_path, encoding='utf-8', sep=',', dtype=str,escapechar='\\')
    calc = MetalnessCalculator(metal_data,control_data)
    while True:
        user_input = input("Enter or a string or path to test: ")
        if type(user_input) == int:
            break
        elif path.exists(str(user_input)):
            with open(user_input, 'r') as file:
                data = file.read().replace('\n', '')
            print(calc.calculate_metalness_score(data))
        else:
            print(calc.calculate_metalness_score(user_input))
    """Input loop Exited. Execution stoped."""

if __name__ == "__main__":
    main()
