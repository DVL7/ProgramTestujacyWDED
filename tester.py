import argparse
import json
import math
import re
import subprocess
import sys
import time
from io import StringIO
from pathlib import Path
import pandas as pd
from pandas.api.types import is_numeric_dtype


DEFAULT_CONFIG = {
    "raw_csv": "iris.csv",
    "program": "nazwa.py",
    "output_csv": "wygenerowane_dane.csv",
    "raw_separator": None,
    "output_separator": None,
    "program_args": [],
    "invoke_with_python": True,
    "output_source": "csv_file",
}

def read_csv_auto(path, separator=None):
    if separator is not None:
        data = pd.read_csv(path, sep=separator)
    else:
        data = pd.read_csv(path)
        if len(data.columns) == 1:
            data = pd.read_csv(path, sep=";")
    return restore_index_if_needed(data)

def read_csv_auto_from_text(text, separator=None):
    data = pd.read_csv(StringIO(text), sep=separator)
    if separator is None and len(data.columns) == 1:
        data = pd.read_csv(StringIO(text), sep=";")
    return restore_index_if_needed(data)

def load_config(config_path):
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku konfiguracyjnego: {config_file}")

    with config_file.open("r", encoding="utf-8") as file:
        loaded_config = json.load(file)

    if not isinstance(loaded_config, dict):
        raise ValueError("Plik konfiguracyjny musi zawierać obiekt JSON.")

    config = DEFAULT_CONFIG.copy()
    config.update(loaded_config)
    return config, config_file.parent

def resolve_path(base_dir, value):
    path = Path(value)

    if not path.is_absolute():
        path = base_dir / path

    return path

def restore_index_if_needed(data):
    if len(data.columns) == 0:
        return data
    first_column = str(data.columns[0])
    if first_column.startswith("Unnamed"):
        values = data.iloc[:, 0]
        if values.is_unique and values.notna().all():
            data = data.set_index(data.columns[0])
            data.index.name = None
    return data

def validate_raw_data(data):
    errors = []

    if data.empty:
        errors.append("Surowe dane wejściowe są puste.")

    if len(data.columns) < 2:
        errors.append("Surowe dane muszą mieć co najmniej jedną kolumnę warunkową i jedną decyzyjną.")

    if data.isna().any().any():
        errors.append("Surowe dane zawierają wartości puste lub brakujące.")

    if len(data.columns) >= 2:
        for column in data.columns[:-1]:
            if not is_numeric_dtype(data[column]):
                try:
                    data[column] = pd.to_numeric(data[column])
                except Exception:
                    errors.append(f"Kolumna warunkowa '{column}' w surowych danych nie jest liczbowa.")

    return errors

def validate_discretized_data(data):
    errors = []

    if data.empty:
        errors.append("Dane zdyskretyzowane są puste.")

    if data.isna().any().any():
        errors.append("Dane zdyskretyzowane zawierają wartości puste lub brakujące.")

    return errors

def parse_boundary(value):
    text = str(value).strip()
    low = text.lower()

    if low in ["-inf", "-infinity"]:
        return -math.inf

    if low in ["inf", "+inf", "infinity", "+infinity"]:
        return math.inf

    return float(text)

def parse_interval(value):
    text = str(value).strip()

    if not ((text.startswith("(") or text.startswith("[")) and (text.endswith(")") or text.endswith("]"))):
        raise ValueError(f"Przedział powinien być zapisany jako tuple, np. (-inf, 5.75), a jest: {value}")

    match = re.fullmatch(r"[\(\[]\s*([^,;]+)\s*[,;]\s*([^)\]]+)\s*[\)\]]", text)

    if not match:
        raise ValueError(f"Niepoprawny format przedziału: {value}")

    left = parse_boundary(match.group(1))
    right = parse_boundary(match.group(2))

    if left >= right:
        raise ValueError(f"Niepoprawny przedział, lewa granica nie jest mniejsza od prawej: {value}")

    return left, right

def validate_basic_structure(raw_data, discretized_data):
    errors = []

    if raw_data.shape[0] != discretized_data.shape[0]:
        errors.append(f"Liczba wierszy się nie zgadza: surowe={raw_data.shape[0]}, zdyskretyzowane={discretized_data.shape[0]}.")

    if len(raw_data.columns) != len(discretized_data.columns):
        errors.append(f"Liczba kolumn się nie zgadza: surowe={len(raw_data.columns)}, zdyskretyzowane={len(discretized_data.columns)}.")
        return errors

    if list(raw_data.columns) != list(discretized_data.columns):
        errors.append("Nazwy lub kolejność kolumn nie są takie same.")
        errors.append(f"Surowe kolumny: {list(raw_data.columns)}")
        errors.append(f"Zdyskretyzowane kolumny: {list(discretized_data.columns)}")

    if list(raw_data.index) != list(discretized_data.index):
        errors.append("Indeksy lub kolejność wierszy zostały zmienione.")

    return errors

def validate_decision_column(raw_data, discretized_data):
    errors = []

    decision_column = raw_data.columns[-1]
    raw_decisions = raw_data[decision_column].astype(str).tolist()
    discretized_decisions = discretized_data[decision_column].astype(str).tolist()

    if raw_decisions != discretized_decisions:
        errors.append("Kolumna decyzyjna została zmieniona.")

    return errors

def validate_intervals(raw_data, discretized_data):
    errors = []
    parsed = pd.DataFrame(index=discretized_data.index)

    for column in raw_data.columns[:-1]:
        parsed_values = []

        for index in raw_data.index:
            raw_value = float(raw_data.loc[index, column])
            cell = discretized_data.loc[index, column]

            try:
                left, right = parse_interval(cell)
            except Exception as error:
                errors.append(f"Wiersz {index}, kolumna '{column}': {error}")
                continue

            if not (left <= raw_value < right):
                errors.append(f"Wiersz {index}, kolumna '{column}': wartość {raw_value} nie należy do przedziału {cell}.")

            parsed_values.append((left, right))

        if len(parsed_values) == len(raw_data.index):
            parsed[column] = parsed_values

    parsed[raw_data.columns[-1]] = discretized_data[raw_data.columns[-1]].to_numpy()

    return errors, parsed

def validate_boundary_values(parsed_data):
    errors = []

    for column in parsed_data.columns[:-1]:
        intervals = set(parsed_data[column].tolist())
        has_minus_inf = any(left == -math.inf for left, right in intervals)
        has_inf = any(right == math.inf for left, right in intervals)

        if not has_minus_inf:
            errors.append(f"Kolumna '{column}' nie ma przedziału zaczynającego się od -inf.")

        if not has_inf:
            errors.append(f"Kolumna '{column}' nie ma przedziału kończącego się na inf.")

    return errors

def find_conflict_groups(discretized_data):
    decision_column = discretized_data.columns[-1]
    conditional_columns = list(discretized_data.columns[:-1])
    groups = []

    grouped = discretized_data.groupby(conditional_columns, sort=False, dropna=False)

    for _, group in grouped:
        if group[decision_column].nunique(dropna=False) > 1:
            groups.append(list(group.index))

    return groups

def count_conflicting_rows(discretized_data):
    return sum(len(group) for group in find_conflict_groups(discretized_data))

def count_intervals(discretized_data):
    total = 0
    details = {}

    for column in discretized_data.columns[:-1]:
        unique_intervals = set(discretized_data[column].tolist())
        details[column] = len(unique_intervals)
        total += len(unique_intervals)

    return total, details

def print_errors(errors):
    print("BŁĘDY:")
    for error in errors:
        print(f"- {error}")

def print_conflict_report(discretized_data):
    groups = find_conflict_groups(discretized_data)

    print("Raport konfliktów:")

    if not groups:
        print("Brak konfliktów.")
    else:
        for group in groups:
            print(", ".join(str(index) for index in group))

    print(f"Liczba konfliktujących wierszy: {count_conflicting_rows(discretized_data)}")

def print_interval_report(discretized_data):
    total, details = count_intervals(discretized_data)

    print("Raport przedziałów:")

    for column, count in details.items():
        print(f"{column}: {count}")

    print(f"Łączna liczba przedziałów: {total}")

def run_discretizer(program_path, raw_path, output_path, program_args, invoke_with_python):
    command = []

    if invoke_with_python and program_path.suffix == ".py":
        command.append(sys.executable)

    command.append(str(program_path))
    command.append(str(raw_path))

    if output_path is not None:
        command.append(str(output_path))

    if program_args:
        command.extend(str(argument) for argument in program_args)

    start_time = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True)
    elapsed_time = time.perf_counter() - start_time

    return completed, elapsed_time

def load_discretized_data_from_run(config, base_dir, completed_process):
    output_source = str(config.get("output_source", "csv_file"))
    output_separator = config.get("output_separator")

    if output_source == "stdout_csv":
        stdout_text = completed_process.stdout.strip()

        if not stdout_text:
            raise ValueError("Program dyskretyzujący nie wypisał żadnych danych na stdout.")

        return read_csv_auto_from_text(stdout_text, output_separator)

    output_path = resolve_path(base_dir, config["output_csv"])

    if not output_path.exists():
        raise FileNotFoundError(f"Program dyskretyzujący nie utworzył pliku wynikowego: {output_path}")

    return read_csv_auto(output_path, output_separator)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="tester_config.json")
    args = parser.parse_args()

    try:
        config, config_dir = load_config(args.config)
    except Exception as error:
        print(f"BŁĄD KONFIGURACJI: {error}")
        return

    raw_csv_path = resolve_path(config_dir, config["raw_csv"])
    program_path = resolve_path(config_dir, config["program"])
    output_source = str(config.get("output_source", "csv_file"))

    if output_source not in {"csv_file", "stdout_csv"}:
        print(f"BŁĄD KONFIGURACJI: nieobsługiwany output_source: {output_source}")
        return

    if not raw_csv_path.exists():
        print(f"Nie znaleziono pliku z surowymi danymi: {raw_csv_path}")
        return

    if not program_path.exists():
        print(f"Nie znaleziono programu dyskretyzującego: {program_path}")
        return

    raw_data = read_csv_auto(raw_csv_path, config.get("raw_separator"))

    output_path = None
    if output_source == "csv_file":
        output_path = resolve_path(config_dir, config["output_csv"])
        output_path.unlink(missing_ok=True)

    try:
        completed_process, elapsed_time = run_discretizer(
            program_path=program_path,
            raw_path=raw_csv_path,
            output_path=output_path,
            program_args=config.get("program_args", []),
            invoke_with_python=bool(config.get("invoke_with_python", True)),
        )
    except Exception as error:
        print(f"BŁĄD URUCHOMIENIA PROGRAMU: {error}")
        return

    errors = []

    if completed_process.returncode != 0:
        errors.append(f"Program dyskretyzujący zakończył się kodem {completed_process.returncode}.")
        if completed_process.stderr:
            errors.append(completed_process.stderr.strip())

    discretized_data = None

    if not errors:
        try:
            discretized_data = load_discretized_data_from_run(config, config_dir, completed_process)
        except Exception as error:
            errors.append(str(error))

    errors.extend(validate_raw_data(raw_data))
    if discretized_data is not None:
        errors.extend(validate_discretized_data(discretized_data))
        errors.extend(validate_basic_structure(raw_data, discretized_data))

    parsed_discretized_data = None

    if not errors and discretized_data is not None:
        errors.extend(validate_decision_column(raw_data, discretized_data))
        interval_errors, parsed_discretized_data = validate_intervals(raw_data, discretized_data)
        errors.extend(interval_errors)

    if not errors and parsed_discretized_data is not None:
        errors.extend(validate_boundary_values(parsed_discretized_data))

    print("Wynik testu:")
    print(f"Czas wykonania programu dyskretyzującego: {elapsed_time:.6f} s")

    if errors:
        print("NIEZALICZONE")
        print_errors(errors)
        return

    print("ZALICZONE")
    print("Struktura danych jest poprawna.")
    print("Indeksy i kolejność wierszy są zachowane.")
    print("Kolumna decyzyjna jest zachowana.")
    print("Przedziały są zapisane jako tuple.")
    print("Wartości brzegowe -inf oraz inf są użyte.")
    print("Wszystkie wartości należą do swoich przedziałów.")
    print_interval_report(parsed_discretized_data)
    print_conflict_report(parsed_discretized_data)

if __name__ == "__main__":
    main()