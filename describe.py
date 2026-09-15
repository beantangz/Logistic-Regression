import csv
import sys
import math

# charge les data dans une list de dict (chaque eleve/ligne a son dict)
def load_dataset(filename: str) -> list[dict[str, str]]:
    with open(filename, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader)

# renvoie True si la data est un chiffre, False sinon (Fist name par exemple)
def is_numeric_column(
    dataset: list[dict[str, str]],
    column: str
) -> bool:
    for row in dataset:
        value = row[column]

        if value == "":
            continue

        try:
            float(value)
        except ValueError:
            return False

    return True

# si collone numerique (True fonction d'avant) -> creer une list de la collonne pour calculs
def get_numeric_values(
    dataset: list[dict[str, str]],
    column: str
) -> list[float]:
    values = []

    for row in dataset:
        value = row[column]

        if value != "":
            values.append(float(value))

    return values


def ft_count(values: list[float]) -> int:
    count = 0

    for _ in values:
        count += 1

    return count

# moyenne
def ft_mean(values: list[float]) -> float:
    total = 0.0

    for value in values:
        total += value

    return total / ft_count(values)


def ft_min(values: list[float]) -> float:
    minimum = values[0]

    for value in values:
        if value < minimum:
            minimum = value

    return minimum


def ft_max(values: list[float]) -> float:
    maximum = values[0]

    for value in values:
        if value > maximum:
            maximum = value

    return maximum

# ecaty type
def ft_std(values: list[float]) -> float:
    count = ft_count(values)

    if count < 2:
        return 0.0

    mean = ft_mean(values)
    squared_difference_sum = 0.0

    for value in values:
        difference = value - mean
        squared_difference_sum += difference * difference

    variance = squared_difference_sum / (count - 1)

    return math.sqrt(variance)

# 25% veut dire 1/4 des valeurs est sous cette valeur
# -> on prend 1/4 de la distance entre l'indice 1/4 et celui du dessus
def ft_percentile(
    sorted_values: list[float],
    percentile: float
) -> float:
    count = ft_count(sorted_values)

    if count == 0:
        return 0.0

    #revele l'indice exact que devrait avoir le 25% (sera surement decimal)
    position = (count - 1) * percentile 

    lower_index = int(position)
    upper_index = lower_index + 1

    if upper_index >= count:
        return sorted_values[lower_index]

    decimal_part = position - lower_index
    #comme position est surement decimal (par exemple 3.7), 
    # et qu'il n'existe pas d'indice 3,7
    # on prend l'indice 3 (lower index) 
    # et on lui ajoute 0.7 (position - lower_index)
    # de la distance jusqua l'indice suivant (upper_value - lower_value) dans calcul final
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]

    # exemple pour 20 indice[1,25] et 30 indice[2] : 
    #           20     +  0.25        * (    30      -      20    ) -> 22.5
    return lower_value + decimal_part * (upper_value - lower_value)


def get_statistics(values: list[float]) -> dict[str, float]:
    # utiliser sorted 1 seul fois qui sert pour chaque calcul de percentile
    sorted_values = sorted(values)

    return {
        "Count": float(ft_count(values)),
        "Mean": ft_mean(values),
        "Std": ft_std(values),
        "Min": ft_min(values),
        "25%": ft_percentile(sorted_values, 0.25),
        "50%": ft_percentile(sorted_values, 0.50),
        "75%": ft_percentile(sorted_values, 0.75),
        "Max": ft_max(values),
    }

def display_results(
    results: dict[str, dict[str, float]]
) -> None:
    statistics_names = [
        "Count",
        "Mean",
        "Std",
        "Min",
        "25%",
        "50%",
        "75%",
        "Max",
    ]

    columns = list(results.keys())
    features_per_table = 4
    label_width = 10

    for start in range(0, len(columns), features_per_table):
        current_columns = columns[
            start:start + features_per_table
        ]

        widths = {}

        for column in current_columns:
            widths[column] = max(len(column) + 2, 16)

        print(" " * label_width, end="")

        for column in current_columns:
            print(f"{column:>{widths[column]}}", end="")

        print()

        for statistic in statistics_names:
            print(f"{statistic:<{label_width}}", end="")

            for column in current_columns:
                value = results[column][statistic]
                width = widths[column]

                print(f"{value:>{width}.6f}", end="")

            print()

        print()

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python3 describe.py <dataset.csv>")
        return

    try:
        dataset = load_dataset(sys.argv[1])
    except FileNotFoundError:
        print(f"Error: file '{sys.argv[1]}' not found")
        return

    if not dataset:
        print("Error: dataset is empty")
        return

    results = {}

    for column in dataset[0]:
        if column == "Index":
            continue

        if is_numeric_column(dataset, column):
            values = get_numeric_values(dataset, column)
            results[column] = get_statistics(values)

    display_results(results)



if __name__ == "__main__":
    main()
