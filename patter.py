import argparse
import io
import importlib.util
import math
import re
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd


AVAILABLE_ALGORITHMS = ["z1.py", "z2.py", "z5.py", "z6.py"]
MAX_ERRORS = 8


def load_discretize_function(module_path: Path):
	spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
	if spec is None or spec.loader is None:
		raise ImportError(f"Cannot load module: {module_path}")

	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)

	if not hasattr(module, "discretize"):
		raise AttributeError(f"{module_path.name} has no discretize(df) function")
	return module.discretize


def parse_boundary(value):
	text = str(value).strip().lower()
	if text in {"-inf", "-infinity"}:
		return -math.inf
	if text in {"inf", "+inf", "infinity", "+infinity"}:
		return math.inf
	return float(value)


def parse_interval(cell):
	def validate_bounds(left, right, source):
		if left >= right:
			raise ValueError(f"invalid interval {source}")
		return left, right

	if isinstance(cell, (tuple, list)) and len(cell) == 2:
		left = parse_boundary(cell[0])
		right = parse_boundary(cell[1])
		return validate_bounds(left, right, cell)

	text = str(cell).strip()
	match = re.fullmatch(r"[\(\[]\s*([^,;]+)\s*[,;]\s*([^\]\)]+)\s*[\)\]]", text)
	if not match:
		raise ValueError(f"invalid interval format: {cell}")
	left = parse_boundary(match.group(1))
	right = parse_boundary(match.group(2))
	return validate_bounds(left, right, cell)


def drop_optional_index_column(raw_df: pd.DataFrame, out_df: pd.DataFrame):
	if len(out_df.columns) == len(raw_df.columns):
		return out_df

	if len(out_df.columns) == len(raw_df.columns) + 1:
		first_col = out_df.columns[0]
		if first_col not in raw_df.columns:
			return out_df.iloc[:, 1:].copy()

	raise ValueError(
		f"column count mismatch: raw={len(raw_df.columns)} out={len(out_df.columns)}"
	)


def count_conflicting_rows(df: pd.DataFrame):
	decision_col = df.columns[-1]
	cond_cols = list(df.columns[:-1])
	grouped = df.groupby(cond_cols, sort=False, dropna=False)
	return int(sum(len(group) for _, group in grouped if group[decision_col].nunique(dropna=False) > 1))


def validate_discretized(raw_df: pd.DataFrame, out_df: pd.DataFrame):
	errors = []
	parsed = pd.DataFrame(index=out_df.index)

	if len(out_df) != len(raw_df):
		errors.append(f"row count mismatch: raw={len(raw_df)} out={len(out_df)}")
		return errors, parsed

	if list(out_df.columns) != list(raw_df.columns):
		errors.append("column names/order mismatch")
		return errors, parsed

	decision_col = raw_df.columns[-1]
	if raw_df[decision_col].astype(str).tolist() != out_df[decision_col].astype(str).tolist():
		errors.append("decision column changed")

	for col in raw_df.columns[:-1]:
		parsed_values = []
		for i, (raw_value, cell) in enumerate(zip(raw_df[col].tolist(), out_df[col].tolist()), start=1):
			try:
				left, right = parse_interval(cell)
			except Exception as exc:
				errors.append(f"{col} row {i}: {exc}")
				if len(errors) >= MAX_ERRORS:
					return errors, parsed
				continue

			numeric = float(raw_value)
			if not (left <= numeric < right):
				errors.append(f"{col} row {i}: value {numeric} not in interval {cell}")
				if len(errors) >= MAX_ERRORS:
					return errors, parsed
			parsed_values.append((left, right))

		if len(parsed_values) == len(raw_df):
			parsed[col] = parsed_values

	parsed[decision_col] = out_df[decision_col].to_numpy()

	for col in parsed.columns[:-1]:
		unique_intervals = set(parsed[col].tolist())
		has_minus_inf = any(left == -math.inf for left, _ in unique_intervals)
		has_inf = any(right == math.inf for _, right in unique_intervals)
		if not has_minus_inf:
			errors.append(f"{col}: missing -inf lower bound")
		if not has_inf:
			errors.append(f"{col}: missing inf upper bound")

	return errors, parsed


def run_case(base_dir: Path, algo_name: str, dataset_name: str):
	algo_path = base_dir / algo_name
	data_path = base_dir / dataset_name

	raw_df = pd.read_csv(data_path)
	discretize = load_discretize_function(algo_path)

	start = time.perf_counter()
	with redirect_stdout(io.StringIO()):
		out_df = discretize(raw_df.copy())
	elapsed = time.perf_counter() - start

	if not isinstance(out_df, pd.DataFrame):
		raise TypeError("discretize() must return pandas.DataFrame")

	normalized = drop_optional_index_column(raw_df, out_df)
	errors, parsed = validate_discretized(raw_df, normalized)

	cut_count = None
	conflict_count = None
	if parsed.shape[0] == raw_df.shape[0] and not errors:
		cut_count = sum(max(parsed[col].nunique() - 1, 0) for col in parsed.columns[:-1])
		conflict_count = count_conflicting_rows(parsed)

	return elapsed, errors, cut_count, conflict_count


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"algorithm",
		help="Algorithm file to test (e.g. z1.py or z1)",
	)
	parser.add_argument(
		"dataset",
		help="Input CSV file to test (e.g. data0.csv)",
	)
	args = parser.parse_args()

	base_dir = Path(__file__).resolve().parent

	selected_algorithm = args.algorithm.strip()
	if not selected_algorithm.endswith(".py"):
		selected_algorithm = f"{selected_algorithm}.py"

	if selected_algorithm not in AVAILABLE_ALGORITHMS:
		print("Unknown algorithm. Allowed values:", ", ".join(AVAILABLE_ALGORITHMS))
		sys.exit(2)

	selected_dataset = args.dataset.strip()
	data_path = base_dir / selected_dataset
	if not data_path.exists():
		print(f"Dataset not found: {selected_dataset}")
		sys.exit(2)

	try:
		elapsed, errors, cut_count, conflict_count = run_case(base_dir, selected_algorithm, selected_dataset)
	except Exception as exc:
		print(f"error: {exc}")
		sys.exit(1)

	if errors:
		print("error: validation failed")
		for err in errors[:5]:
			print(f"- {err}")
		sys.exit(1)

	print(f"czas_wykonania_s: {elapsed:.6f}")
	print(f"liczba_ciec: {cut_count}")
	print(f"konflikty: {conflict_count}")


if __name__ == "__main__":
	main()
