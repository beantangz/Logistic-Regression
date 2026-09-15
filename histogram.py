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

# Creer une list des notes de chaque eleves pour une maison et un cours choisit en param
def get_house_scores(
    dataset: list[dict[str, str]],
    course: str,
    house: str
) -> list[float]:
    scores = []

    for student in dataset:
        if student["Hogwarts House"] != house:
            continue

        value = student[course]

        if value == "":
            continue

        scores.append(float(value))

    return scores



def display_course_histogram(
    axis,
    dataset: list[dict[str, str]],
    course: str
) -> None:
    all_scores = []

    # boucle pour l'echelle des histogrammes-> donne score_range
    for house in HOUSES:
        house_scores = get_house_scores(
            dataset,
            course,
            house
        )
        all_scores.extend(house_scores)

    score_range = (
        min(all_scores),
        max(all_scores)
    )

    # boucle d'affichage chaque maison avec sa couleur et son score associe
    for house in HOUSES:
        house_scores = get_house_scores(
            dataset,
            course,
            house
        )

        axis.hist(
            house_scores,
            bins=20,
            range=score_range,
            density=True,
            alpha=0.45,
            color=COLORS[house],
            label=house,
        )

    axis.set_title(f"Distribution of {course}")
    axis.set_xlabel("Score")
    axis.set_ylabel("Density")
    axis.grid(alpha=0.2)


def display_all_histograms(
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
        4,
        4,
        figsize=(18, 14)
    )

    flat_axes = axes.flatten()

    for index, course in enumerate(courses):
        display_course_histogram(
            flat_axes[index],
            dataset,
            course
        )

    for index in range(len(courses), len(flat_axes)):
        flat_axes[index].set_visible(False)

    handles, labels = flat_axes[0].get_legend_handles_labels()

    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4
    )

    figure.suptitle(
        "Course distributions by Hogwarts house",
        fontsize=18
    )

    figure.tight_layout(
        rect=(0, 0, 1, 0.94)
    )

    plt.show()

def main() -> None:
    if len(sys.argv) != 2:
        print(
            "Usage: python3 histogram.py "
            "<dataset.csv>"
        )
        return

    try:
        dataset = load_dataset(sys.argv[1])

        if not dataset:
            print("Error: dataset is empty")
            return

        display_all_histograms(dataset)

    except (OSError, csv.Error, ValueError) as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()