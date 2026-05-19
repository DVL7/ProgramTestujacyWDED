from bisect import bisect_right, bisect_left
from collections import defaultdict
import pandas as pd


def load_data(filename):
    df = pd.read_csv(filename)
    return df


def condition_columns(df):
    ignored_cols = {'index', df.columns[-1]}
    return [col for col in df.columns if col not in ignored_cols]


def decision_column(df):
    return df.columns[-1]


def generate_cuts(df):
    decision = decision_column(df)
    cuts = []

    for column in condition_columns(df):
        grouped = df.groupby(column)[decision].agg(lambda x: set(x))
        values = sorted(grouped.index)

        for left, right in zip(values, values[1:]):
            if grouped[left] != grouped[right]:
                cut = (left + right) / 2
                cuts.append((column, cut))

    return cuts


def prepare_cuts(cuts):
    return sorted(cuts, key=lambda x: (x[0], x[1]))


def cuts_by_attribute(cuts):
    values = defaultdict(list)
    indexes = defaultdict(list)

    for i, (column, cut) in enumerate(cuts):
        values[column].append(cut)
        indexes[column].append(i)

    return values, indexes


def greedy_remove_cuts(df):
    cuts = prepare_cuts(generate_cuts(df))
    attributes = condition_columns(df)
    decision = decision_column(df)
    values_by_attr, indexes_by_attr = cuts_by_attribute(cuts)

    pair_to_cuts = []
    cut_to_pairs = [[] for _ in cuts]

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if df.iloc[i][decision] == df.iloc[j][decision]:
                continue

            separating = []

            for attribute in attributes:
                v1 = float(df.iloc[i][attribute])
                v2 = float(df.iloc[j][attribute])

                low = min(v1, v2)
                high = max(v1, v2)

                values = values_by_attr[attribute]
                indexes = indexes_by_attr[attribute]

                left = bisect_right(values, low)
                right = bisect_left(values, high)

                separating.extend(indexes[left:right])

            if separating:
                pair_index = len(pair_to_cuts)
                pair_to_cuts.append(separating)

                for cut_index in separating:
                    cut_to_pairs[cut_index].append(pair_index)

    active = [True] * len(cuts)
    coverage = [len(x) for x in pair_to_cuts]
    unique = [0] * len(cuts)

    for pair in pair_to_cuts:
        if len(pair) == 1:
            unique[pair[0]] += 1

    while True:
        removable = []

        for i in range(len(cuts)):
            if active[i] and unique[i] == 0:
                removable.append(i)

        if not removable:
            break

        worst = min(removable, key=lambda x: len(cut_to_pairs[x]))
        active[worst] = False

        for pair_index in cut_to_pairs[worst]:
            if coverage[pair_index] == 2:
                for other in pair_to_cuts[pair_index]:
                    if active[other] and other != worst:
                        unique[other] += 1
                        break

            coverage[pair_index] -= 1

    result = []
    for i in range(len(cuts)):
        if active[i]:
            result.append(cuts[i])

    return result


def interval_for_value(value, cuts):
    pos = bisect_right(cuts, value)

    if pos == 0:
        left = float("-inf")
    else:
        left = cuts[pos - 1]

    if pos == len(cuts):
        right = float("inf")
    else:
        right = cuts[pos]

    return (left, right)


def discretize(df):
    result = df.copy()

    if 'index' not in result.columns:
        result.insert(0, 'index', range(1, len(result) + 1))

    cuts_dict = defaultdict(list)
    for column, cut in greedy_remove_cuts(df):
        cuts_dict[column].append(cut)

    for column in condition_columns(df):
        column_cuts = sorted(cuts_dict[column])
        result[column] = result[column].apply(
            lambda x: interval_for_value(float(x), column_cuts)
        )

    return result


def main():
    filename = input("Podaj nazwę pliku CSV (np. iris.csv): ")

    df = load_data(filename)
    discretized = discretize(df)

    cols = ['index'] + [col for col in discretized.columns if col != 'index']
    discretized = discretized[cols]

    discretized.to_csv("discretized.csv", index=False)
    print("Zapisano discretized.csv")


if __name__ == "__main__":
    main()