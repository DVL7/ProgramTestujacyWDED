import argparse
from pathlib import Path

import pandas as pd


BIN_MAP = {
    "sepal.length": [
        (-float("inf"), 5.1, "(-inf, 5.1)"),
        (5.1, 5.8, "[5.1, 5.8)"),
        (5.8, float("inf"), "[5.8, inf)"),
    ],
    "sepal.width": [
        (-float("inf"), 2.8, "(-inf, 2.8)"),
        (2.8, 3.2, "[2.8, 3.2)"),
        (3.2, float("inf"), "[3.2, inf)"),
    ],
    "petal.length": [
        (-float("inf"), 2.5, "(-inf, 2.5)"),
        (2.5, 4.9, "[2.5, 4.9)"),
        (4.9, float("inf"), "[4.9, inf)"),
    ],
    "petal.width": [
        (-float("inf"), 0.8, "(-inf, 0.8)"),
        (0.8, 1.8, "[0.8, 1.8)"),
        (1.8, float("inf"), "[1.8, inf)"),
    ],
}


def discretize_value(value, bins):
    numeric_value = float(value)

    for left, right, label in bins:
        if left <= numeric_value < right:
            return label

    raise ValueError(f"Nie udało się przypisać wartości {value} do przedziału.")


def discretize_frame(frame):
    result = frame.copy()

    for column in result.columns[:-1]:
        if column not in BIN_MAP:
            raise KeyError(f"Brak reguł dyskretyzacji dla kolumny: {column}")
        result[column] = result[column].map(lambda value, bins=BIN_MAP[column]: discretize_value(value, bins))

    return result


def main():
    parser = argparse.ArgumentParser(description="Testowy program dyskretyzujący iris.csv.")
    parser.add_argument("input_csv", help="Ścieżka do surowego pliku CSV")
    parser.add_argument("output_csv", help="Ścieżka do pliku wynikowego CSV")
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)

    data = pd.read_csv(input_path)
    discretized = discretize_frame(data)
    discretized.to_csv(output_path, index=True)


if __name__ == "__main__":
    main()
