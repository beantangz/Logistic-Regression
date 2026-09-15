import csv
import sys

import matplotlib.pyplot as plt


HOUSES = [
    "Gryffindor",
    "Hufflepuff",
    "Ravenclaw",
    "Slytherin",
]

COLORS = {
    "Gryffindor": "red",
    "Hufflepuff": "gold",
    "Ravenclaw": "blue",
    "Slytherin": "green",
}


def load_dataset(filename: str) -> list[dict[str, str]]:
    with open(filename, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader)


def get_house_points(
    dataset: list[dict[str, str]],
    house: str,
    x_feature: str,
    y_feature: str
) -> tuple[list[float], list[float]]:
    x_values = []
    y_values = []

    for student in dataset:
        if student["Hogwarts House"] != house:
            continue

        x_value = student[x_feature]
        y_value = student[y_feature]

        if x_value == "" or y_value == "":
            continue

        x_values.append(float(x_value))
        y_values.append(float(y_value))

    return x_values, y_values


def display_scatter_plot(
    dataset: list[dict[str, str]],
    x_feature: str,
    y_feature: str
) -> None:
    figure, axis = plt.subplots(
        figsize=(10, 7)
    )

    for house in HOUSES:
        x_values, y_values = get_house_points(
            dataset,
            house,
            x_feature,
            y_feature
        )

        axis.scatter(
            x_values,
            y_values,
            color=COLORS[house],
            label=house, #pour la legende
            alpha=0.55,
            s=18,
        )

    axis.set_title(
        f"{x_feature} vs {y_feature}"
    )
    axis.set_xlabel(x_feature)
    axis.set_ylabel(y_feature)
    axis.legend()
    axis.grid(alpha=0.2)

    figure.tight_layout() #matplotlib organise bien les titres et tt...
    plt.show()


def main() -> None:
    if len(sys.argv) != 2:
        print(
            "Usage: python3 scatter_plot.py "
            "<dataset.csv>"
        )
        return

    try:
        dataset = load_dataset(sys.argv[1])

        if not dataset:
            print("Error: dataset is empty")
            return

        display_scatter_plot( #changer ici pour voir correlation d'autres matieres
            dataset,
            "Astronomy",
            "Defense Against the Dark Arts"
        )

    except (OSError, csv.Error, ValueError, KeyError) as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()