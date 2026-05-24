import io
import importlib.util
import math
import re
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd


def load_discretize(algo_path: Path):
	spec = importlib.util.spec_from_file_location(algo_path.stem, algo_path)
	if spec is None or spec.loader is None:
		raise ImportError(f"Nie mozna zaladowac programu: {algo_path}")

	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)

	if not hasattr(module, "discretize"):
		raise AttributeError(f"{algo_path.name} nie ma funkcji discretize(df)")

	return module.discretize


def normalize_output(df: pd.DataFrame, out_df: pd.DataFrame) -> pd.DataFrame:
	if len(out_df.columns) == len(df.columns):
		return out_df

	if len(out_df.columns) == len(df.columns) + 1:
		index_col = out_df.columns[0]
		if index_col not in df.columns or str(index_col).lower() in {"index", "unnamed: 0", "unnamed"}:
			return out_df.iloc[:, 1:].copy()

	raise ValueError(
		f"Niepoprawna liczba kolumn: wejscie={len(df.columns)}, wyjscie={len(out_df.columns)}"
	)


def parse_boundary(value):
	text = str(value).strip().lower()
	if text in {"-inf"}:
		return -math.inf
	if text in {"inf"}:
		return math.inf
	return float(value)


def parse_interval(value):
	if isinstance(value, (tuple, list)) and len(value) == 2:
		left = parse_boundary(value[0])
		right = parse_boundary(value[1])
		if left >= right:
			raise ValueError(f"niepoprawny przedzial: {value}")
		return left, right

	text = str(value).strip()
	match = re.fullmatch(r"[\(\[]\s*([^,;]+)\s*[,;]\s*([^\)\]]+)\s*[\)\]]", text)
	if not match:
		raise ValueError(f"niepoprawny format przedzialu: {value}")
	left = parse_boundary(match.group(1))
	right = parse_boundary(match.group(2))
	if left >= right:
		raise ValueError(f"niepoprawny przedzial: {value}")
	return left, right


def run_test(discretize, df: pd.DataFrame):
	start = time.perf_counter()
	
	out_df = discretize(df.copy())
	
	elapsed = time.perf_counter() - start

	if not isinstance(out_df, pd.DataFrame):
		raise TypeError("discretize() musi zwracac pandas.DataFrame")

	out_df = normalize_output(df, out_df)
	errors = []
	parsed = pd.DataFrame(index=out_df.index)

	if len(out_df) != len(df):
		errors.append(f"niezgodna liczba wierszy: wejscie={len(df)}, wyjscie={len(out_df)}")

	if list(out_df.columns) != list(df.columns):
		errors.append("niezgodne nazwy lub kolejnosc kolumn")

	decision_col = df.columns[-1]
	if df[decision_col].astype(str).tolist() != out_df[decision_col].astype(str).tolist():
		errors.append("kolumna decyzyjna zostala zmieniona")

	for col in df.columns[:-1]:
		parsed_values = []
		for idx, (raw_value, cell) in enumerate(zip(df[col].tolist(), out_df[col].tolist()), start=1):
			try:
				left, right = parse_interval(cell)
			except Exception as exc:
				errors.append(f"wiersz {idx}, kolumna '{col}': {exc}")
				continue

			numeric = float(raw_value)
			if not (left <= numeric < right):
				errors.append(f"wiersz {idx}, kolumna '{col}': wartosc {numeric} nie nalezy do przedzialu {cell}")
			parsed_values.append((left, right))

		if len(parsed_values) == len(df):
			parsed[col] = parsed_values

	parsed[decision_col] = out_df[decision_col].to_numpy()

	for col in parsed.columns[:-1]:
		intervals = set(parsed[col].tolist())
		has_minus_inf = any(left == -math.inf for left, _ in intervals)
		has_inf = any(right == math.inf for _, right in intervals)
		if not has_minus_inf:
			errors.append(f"kolumna '{col}' nie ma przedzialu zaczynajacego sie od -inf")
		if not has_inf:
			errors.append(f"kolumna '{col}' nie ma przedzialu konczacego sie na inf")

	cut_count = sum(max(parsed[col].nunique() - 1, 0) for col in parsed.columns[:-1]) if not errors else None
	conflicts = None
	conflict_groups = []
	if not errors:
		cond_cols = list(parsed.columns[:-1])
		grouped = parsed.groupby(cond_cols, sort=False, dropna=False)
		conflicts = 0
		for _, group in grouped:
			if group[decision_col].nunique(dropna=False) > 1:
				rows = []
				for idx_label in group.index.tolist():
					try:
						pos = int(df.index.get_loc(idx_label)) + 1
					except Exception:
						try:
							pos = int(idx_label) + 1
						except Exception:
							pos = None
					if pos is not None:
						rows.append(pos)
				conflict_groups.append(rows)
				conflicts += len(group)

	return elapsed, cut_count, conflicts, conflict_groups, errors


def main():
	if len(sys.argv) != 3:
		print("Uzycie: python ztest.py plik.py data.csv")
		sys.exit(2)

	algo_file = sys.argv[1]
	data_file = sys.argv[2]

	algo_path = Path(algo_file)
	data_path = Path(data_file)

	if not algo_path.exists():
		print(f"Blad: nie znaleziono pliku algorytmu: {algo_file}")
		sys.exit(1)

	if algo_path.suffix != ".py":
		print("Blad: pierwszy argument musi byc plikiem .py")
		sys.exit(1)

	if not data_path.exists():
		print(f"Blad: nie znaleziono pliku danych: {data_file}")
		sys.exit(1)

	if data_path.suffix != ".csv":
		print("Blad: drugi argument musi byc plikiem .csv")
		sys.exit(1)


	df = pd.read_csv(data_path)
	discretize = load_discretize(algo_path)

	try:
		elapsed, cut_count, conflicts, conflict_groups, errors = run_test(discretize, df)
	except Exception as exc:
		print(f"Blad: {exc}")
		sys.exit(1)

	print("Wynik testu:")
	print(f"Czas wykonania programu dyskretyzujacego: {elapsed:.6f} s")
	if errors:
		print("NIEZALICZONE")
		for error in errors:
			print(f"- {error}")
		sys.exit(1)

	print("ZALICZONE")
	print(f"Liczba cięć: {cut_count}")
	print(f"Konflikty: {conflicts}")
	print("Raport Konfliktów:")
	seen = set()
	for group in conflict_groups:
		to_print = [r for r in group if r not in seen]
		if not to_print:
			continue
		print(", ".join(str(r) for r in to_print))
		for r in to_print:
			seen.add(r)

if __name__ == "__main__":
	main()
