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


def display_all_scatter_plots(
    dataset: list[dict[str, str]]
) -> None:
    courses = [
        "Arithmancy",
        "Astronomy",
        "Herbology",
        "Defense Against the Dark Arts",
        "Divination",
        "Muggle Studies",
        "Ancient Runes",
        "History of Magic",
        "Transfiguration",
        "Potions",
        "Care of Magical Creatures",
        "Charms",
        "Flying",
    ]

    figure, axes = plt.subplots(
        len(courses),
        len(courses),
        figsize=(25, 25)
    )

    for row, y_feature in enumerate(courses):
        for column, x_feature in enumerate(courses):
            axis = axes[row][column]

            if row == column:
                axis.text(
                    0.5,
                    0.5,
                    x_feature,
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                    fontsize=7,
                )
            else:
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
                        alpha=0.35,
                        s=3,
                    )

            axis.set_xticks([])
            axis.set_yticks([])

    figure.suptitle(
        "Scatter plot matrix",
        fontsize=20
    )
    figure.tight_layout(
        rect=(0, 0, 1, 0.98)
    )

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

        display_all_scatter_plots(dataset)

    except (OSError, csv.Error, ValueError, KeyError) as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()