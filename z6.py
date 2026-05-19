import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

def discretize(df: pd.DataFrame) -> pd.DataFrame:
    decision_col = df.columns[-1]
    attr_cols = list(df.columns[:-1])
    n = len(df)
    if n == 0 or len(attr_cols) == 0:
        return df.copy()

    # zamieniamy etykiety decyzji (np. "good"/"bad") na liczby 0,1,2...
    # łatwiej potem liczyć na tablicach numpy
    _, decisions = np.unique(df[decision_col].to_numpy(), return_inverse=True)
    decisions = decisions.astype(np.int32)
    unique_classes = np.unique(decisions)
    num_classes = len(unique_classes)

    # ile obiektów należy do każdej klasy
    # + ile łącznie jest par obiektów z różnych klas (to nasz cel do rozdzielenia)
    class_counts = np.bincount(decisions)
    total_diff_pairs = 0
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            total_diff_pairs += class_counts[i] * class_counts[j]

    # --- szukanie kandydatów na cięcia ---
    cuts_info = []
    masks = []

    for col in attr_cols:
        vals = df[col].to_numpy(dtype=np.float32)
        order = np.argsort(vals, kind='mergesort')
        sv = vals[order]
        sd = decisions[order]

        # przechodzimy po posortowanych wartościach
        # grupujemy obiekty z tą samą wartością atrybutu (nie można ciąć w środku grupy)
        i = 0
        while i < n - 1:
            j = i
            while j + 1 < n and sv[j + 1] == sv[i]:
                j += 1
            if j < n - 1:
                # cięcie ma sens tylko jeśli po obu stronach są różne klasy
                if sv[j] != sv[j + 1] and not np.all(sd[i:j+1] == sd[j+1]):
                    cut_val = (sv[j] + sv[j+1]) / 2.0
                    cuts_info.append((col, float(cut_val)))
                    masks.append(vals < cut_val)
            i = j + 1

    n_cuts = len(cuts_info)
    if n_cuts == 0:
        return _build_result(df, attr_cols, decision_col, {c: [] for c in attr_cols})

    # składamy wszystkie maski w jedną macierz (n_cuts x n) - szybszy dostęp później
    masks_arr = np.stack(masks, axis=0)
    del masks

    # --- ile par każde cięcie faktycznie rozdziela (cut_size) ---
    cut_sizes = np.zeros(n_cuts, dtype=np.int32)
    for k in range(n_cuts):
        mask = masks_arr[k]
        below_counts = np.bincount(decisions[mask], minlength=num_classes)
        above_counts = class_counts - below_counts

        # pary, które NIE są rozdzielone = obie po tej samej stronie cięcia
        not_separated = 0
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                not_separated += below_counts[i] * below_counts[j] + above_counts[i] * above_counts[j]

        # cut_size = ile par to cięcie rozdziela
        cut_sizes[k] = total_diff_pairs - not_separated

    # --- generujemy wszystkie pary obiektów z różnych klas ---
    # to są pary, które musimy rozdzielić przynajmniej jednym cięciem
    class_to_indices = {c: np.where(decisions == c)[0] for c in unique_classes}
    pairs_list = []

    for idx_a, c_a in enumerate(unique_classes):
        for idx_b, c_b in enumerate(unique_classes):
            if idx_a >= idx_b: continue

            indices_a = class_to_indices[c_a]
            indices_b = class_to_indices[c_b]

            for i in indices_a:
                for j in indices_b:
                    pairs_list.append((i, j) if i < j else (j, i))

    pairs = np.array(pairs_list, dtype=np.int32)
    n_pairs = len(pairs)
    if n_pairs == 0:
        return _build_result(df, attr_cols, decision_col, {c: [] for c in attr_cols})

    pair_i_arr = pairs[:, 0]
    pair_j_arr = pairs[:, 1]
    del pairs_list, pairs

    # pair_count[p] = przez ile aktywnych cięć para p jest rozdzielona
    # critical_owner[p] = jeśli para ma dokładnie 1 cięcie, to które (indeks)
    pair_count = np.zeros(n_pairs, dtype=np.int32)
    critical_owner = np.full(n_pairs, -1, dtype=np.int32)

    for k in range(n_cuts):
        mask = masks_arr[k]
        sep = (mask[pair_i_arr] != mask[pair_j_arr])
        pair_count[sep] += 1

    # pary z count==1 są "krytyczne" - ich jedyne cięcie nie może być usunięte
    crit = np.where(pair_count == 1)[0]
    for p_idx in crit:
        i, j = pair_i_arr[p_idx], pair_j_arr[p_idx]
        owner = np.argmax(masks_arr[:, i] != masks_arr[:, j])
        critical_owner[p_idx] = owner

    # --- właściwe zachłanne usuwanie ---
    active = np.ones(n_cuts, dtype=bool)
    INF_COST = 2147483647

    while True:
        # cięcia esencjalne = są jedynym właścicielem jakiejś pary, nie ruszamy ich
        essential = np.zeros(n_cuts, dtype=bool)
        valid_owners = critical_owner[critical_owner >= 0]
        if len(valid_owners) > 0:
            essential[valid_owners] = True

        removable = active & (~essential)
        if not removable.any():
            break

        # usuwamy cięcie o najmniejszym cut_size (najmniej użyteczne)
        costs = np.where(removable, cut_sizes, INF_COST)
        best = int(np.argmin(costs))
        active[best] = False

        # aktualizujemy liczniki par, które to cięcie rozdzielało
        mask_best = masks_arr[best]
        affected_mask = (mask_best[pair_i_arr] != mask_best[pair_j_arr])
        affected_indices = np.where(affected_mask)[0]

        old_counts = pair_count[affected_indices].copy()
        pair_count[affected_indices] -= 1

        # pary które spadły z 2 na 1 - stają się krytyczne, szukamy ich nowego właściciela
        newly_critical = affected_indices[(old_counts == 2) & (pair_count[affected_indices] == 1)]
        if len(newly_critical) > 0:
            for p_idx in newly_critical:
                i, j = pair_i_arr[p_idx], pair_j_arr[p_idx]
                sep_cuts = np.where(masks_arr[:, i] != masks_arr[:, j])[0]
                for sc in sep_cuts:
                    if active[sc]:
                        critical_owner[p_idx] = sc
                        break

        # pary które spadły z 1 na 0 - nikt ich już nie rozdziela, czyścimy właściciela
        dead_pairs = affected_indices[(old_counts == 1) & (pair_count[affected_indices] == 0)]
        if len(dead_pairs) > 0:
            critical_owner[dead_pairs] = -1

    # zbieramy aktywne cięcia per atrybut i budujemy wynik
    active_cuts_per_col = {col: [] for col in attr_cols}
    for k in np.where(active)[0]:
        col, val = cuts_info[k]
        active_cuts_per_col[col].append(val)

    return _build_result(df, attr_cols, decision_col, active_cuts_per_col)

def _build_result(df, attr_cols, decision_col, active_cuts_per_col):
    n = len(df)
    result = pd.DataFrame(index=df.index)
    for col in attr_cols:
        col_cuts = np.array(sorted(active_cuts_per_col.get(col, [])), dtype=np.float32)
        vals = df[col].to_numpy(dtype=np.float32)
        if len(col_cuts) == 0:
            # brak cięć = jeden wielki przedział od -inf do inf
            result[col] = [(float('-inf'), float('inf'))] * n
        else:
            # dla każdego obiektu znajdź do którego przedziału należy
            idx = np.searchsorted(col_cuts, vals, side='right')
            L = len(col_cuts)
            left = np.where(idx > 0, col_cuts[np.clip(idx - 1, 0, L - 1)], float('-inf'))
            right = np.where(idx < L, col_cuts[np.clip(idx, 0, L - 1)], float('inf'))
            result[col] = list(zip(left.tolist(), right.tolist()))
    result[decision_col] = df[decision_col].to_numpy()
    result.insert(0, 'index', range(n))
    return result

if __name__ == "__main__":
    try:
        df = pd.read_csv("winequality-red.csv")
        out = discretize(df)
        print(out)
        out.to_csv("wine_red_discretized5.csv")
    except FileNotFoundError:
        print("File winequality-red.csv not found.")
