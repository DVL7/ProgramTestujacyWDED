import pandas as pd
import numpy as np

def load_data(file_path):
    df = pd.read_csv(file_path)
    df.index = range(2, len(df) + 2)

    if df.isnull().values.any():
        raise ValueError("Dane zawierają puste wartości!")

    if len(df.columns) < 2:
        raise ValueError("Plik musi mieć co najmniej 2 kolumny: atrybut warunkowy i decyzja.")

    for col in df.columns[:-1]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Kolumna warunkowa '{col}' musi być liczbowa.")

    return df


def save_data(df, output_path):
    df_save = df.copy()
    for col in df_save.columns[:-1]:
        df_save[col] = df_save[col].apply(lambda x: fmt_interval(x[0], x[1]))
    df_save.to_csv(output_path, index=True, index_label="Indeks")


def fmt_interval(l, r):
    lb = "(" if np.isinf(l) and l < 0 else "["
    rb = ")" if np.isinf(r) and r > 0 else ")"
    return f"{lb}{l}, {r}{rb}"


def find_cut_points(column, decision_column):
    temp_df = pd.DataFrame({
        "value": column,
        "decision": decision_column
    }).sort_values("value").reset_index(drop=True)

    cut_points = []
    group_start = 0
    n = len(temp_df)

    while group_start < n:
        group_decision = temp_df.loc[group_start, "decision"]
        group_end = group_start

        while (
            group_end + 1 < n
            and temp_df.loc[group_end + 1, "decision"] == group_decision
        ):
            group_end += 1

        next_start = group_end + 1

        if next_start < n:
            left_max = temp_df.loc[group_end, "value"]
            right_min = temp_df.loc[next_start, "value"]
            if left_max != right_min:
                cut_points.append((left_max + right_min) / 2)

        group_start = next_start

    return sorted(set(cut_points))


def count_conflicts(combined_base, label_matrix, col_idx, new_col_labels, decision, strides):
    combined = (
        combined_base
        - label_matrix[:, col_idx].astype(np.int64) * strides[col_idx]
        + new_col_labels.astype(np.int64) * strides[col_idx]
    )
    df_tmp = pd.DataFrame({'g': combined, 'd': decision})
    return int((df_tmp.groupby('g')['d'].transform('nunique') > 1).sum())


def count_intervals(discretized_df):
    return sum(
        len(discretized_df[col].unique())
        for col in discretized_df.columns[:-1]
    )

def discretize(df):
    discretized_df = df.copy()
    conditional_columns = list(df.columns[:-1])
    decision_col = df.columns[-1]

    decision_raw = df[decision_col].values
    dec_map = {v: i for i, v in enumerate(pd.unique(decision_raw))}
    decision_int = np.array([dec_map[v] for v in decision_raw], dtype=np.int32)

    all_cut_points = {
        col: find_cut_points(df[col], df[decision_col])
        for col in conditional_columns
    }

    used_cuts = {}
    for col in conditional_columns:
        cuts = all_cut_points[col]
        if cuts:
            used_cuts[col] = [cuts[len(cuts) // 2]]  # środkowy punkt cięcia
        else:
            used_cuts[col] = [float(np.median(df[col].values))]

    label_matrix = np.zeros((len(df), len(conditional_columns)), dtype=np.int32)
    for ci, col in enumerate(conditional_columns):
        label_matrix[:, ci] = np.searchsorted(used_cuts[col], df[col].values, side='right').astype(np.int32)

    def compute_strides(m):
        dims = m.max(axis=0).astype(np.int64) + 2
        s = np.ones(m.shape[1], dtype=np.int64)
        for i in range(m.shape[1] - 2, -1, -1):
            s[i] = s[i + 1] * dims[i + 1]
        return s

    strides = compute_strides(label_matrix)
    combined_base = (label_matrix.astype(np.int64) * strides).sum(axis=1)
    current_conflicts = count_conflicts(combined_base, label_matrix, 0, label_matrix[:, 0], decision_int, strides)

    improvement = True
    while improvement:
        improvement = False
        best_col_idx = None
        best_cut = None
        best_val = current_conflicts
        best_labels = None

        for ci, col in enumerate(conditional_columns):
            col_vals = df[col].values
            for cut in all_cut_points[col]:
                if cut in used_cuts[col]:
                    continue

                new_labels = np.searchsorted(sorted(used_cuts[col] + [cut]), col_vals, side='right').astype(np.int32)
                c = count_conflicts(combined_base, label_matrix, ci, new_labels, decision_int, strides)

                if c < best_val:
                    best_val = c
                    best_col_idx = ci
                    best_cut = cut
                    best_labels = new_labels

        if best_cut is not None:
            col = conditional_columns[best_col_idx]
            used_cuts[col].append(best_cut)
            used_cuts[col].sort()
            label_matrix[:, best_col_idx] = best_labels
            strides = compute_strides(label_matrix)
            combined_base = (label_matrix.astype(np.int64) * strides).sum(axis=1)
            current_conflicts = best_val
            improvement = True

    for ci, col in enumerate(conditional_columns):
        cuts = sorted(used_cuts[col])

        col_min = float(df[col].min())
        if cuts[0] <= col_min:
            unique_vals = sorted(df[col].unique())
            naive_cuts = [(unique_vals[i] + unique_vals[i+1]) / 2 for i in range(len(unique_vals)-1)]
            candidates = all_cut_points[col] if all_cut_points[col] else naive_cuts
            valid = [c for c in candidates if c > col_min]
            if valid:
                # Usuń cięcia <= col_min, dodaj pierwsze cięcie > col_min
                cuts = sorted([c for c in cuts if c > col_min] + [valid[0]])
                used_cuts[col] = cuts

        cuts = [float(round(c, 4)) for c in sorted(cuts)]
        bounds = [-np.inf] + cuts + [np.inf]
        intervals = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
        discretized_df[col] = [next((l, r) for l, r in intervals if l <= v < r) for v in df[col].values]

    return discretized_df


def main():

    file_path = input("Podaj nazwę pliku CSV: ").strip()

    try:
        df = load_data(file_path)
    except FileNotFoundError:
        print(f"Błąd: plik '{file_path}' nie istnieje.")
        return
    except ValueError as e:
        print(f"Błąd danych: {e}")
        return

    discretized_df = discretize(df)

    print(f"\nLiczba przedziałów: {count_intervals(discretized_df)}")

    save_data(discretized_df, "output.csv")

    print("\nKońcowy wynik:")
    display_df = discretized_df.copy()
    for col in display_df.columns[:-1]:
        display_df[col] = display_df[col].apply(lambda x: fmt_interval(x[0], x[1]))
    display_df.index = range(2, len(display_df) + 2)
    display_df.index.name = 'Indeks'
    print(display_df.to_string())


if __name__ == "__main__":
    main()