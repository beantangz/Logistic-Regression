import csv
import json
import sys

import numpy as np


def load_dataset( #charge le dataset de notes -> chaque ligne devient un dict
    filename: str
) -> list[dict[str, str]]:
    with open(filename, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader)


def load_model(filename: str) -> dict: # charge weights.json en dict, par exple : model_data["features"]
    with open(filename, "r", encoding="utf-8") as file:
        return json.load(file)


def prepare_test_features(
    dataset: list[dict[str, str]],
    model_data: dict
) -> np.ndarray:
    features = model_data["features"]
    missing_means = model_data["missing_value_means"]
    rows = []

    for student in dataset:
        row = []

        for index, feature in enumerate(features):
            value = student[feature]

            if value == "":
                row.append(missing_means[index])
            else:
                row.append(float(value))

        rows.append(row)

    return np.array(rows) # creer un tableau de tableau de notes par eleves (matrice)


def normalize_test_features(
    features: np.ndarray,
    model_data: dict
) -> np.ndarray:
    means = np.array(
        model_data["normalization_means"] # recupere les donnes de weights
    )
    standard_deviations = np.array(
        model_data["normalization_stds"]
    )

    return (
        features - means # normalise les notes
    ) / standard_deviations


def sigmoid(value: float) -> float: # transforme valeur en proba entre 0 et 1 pour choix maison
    value = float(np.clip(value, -500, 500))
    return 1.0 / (1.0 + np.exp(-value))


def predict_houses(
    features: np.ndarray,
    model_data: dict
) -> list[str]:
    predictions = []

    for student in features: # pour chaque eleve
        best_house = ""
        best_score = -1.0

        for house in model_data["houses"]: # on calcul le score de chaque maison
            house_model = model_data["models"][house]
            weights = np.array(
                house_model["weights"]
            )
            bias = float(house_model["bias"])

            linear_result = (
                student @ weights
            ) + bias

            score = sigmoid(linear_result)

            if score > best_score: # si meilleur score, on retient la maison et le score
                best_score = score
                best_house = house

        predictions.append(best_house)

    return predictions


def save_predictions(
    predictions: list[str],
    filename: str
) -> None:
    with open(
        filename,
        "w",
        encoding="utf-8",
        newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "Index",
            "Hogwarts House"
        ])

        for index, house in enumerate(predictions):
            writer.writerow([
                index,
                house
            ])


def main() -> None:
    if len(sys.argv) != 3:
        print(
            "Usage: python3 logreg_predict.py "
            "<dataset_test.csv> <weights.json>"
        )
        return

    try:
        dataset = load_dataset(sys.argv[1])
        model_data = load_model(sys.argv[2])

        if not dataset:
            print("Error: dataset is empty")
            return

        features = prepare_test_features(
            dataset,
            model_data
        )

        normalized_features = normalize_test_features(
            features,
            model_data
        )

        predictions = predict_houses(
            normalized_features,
            model_data
        )

        save_predictions(
            predictions,
            "houses.csv"
        )

        print("Predictions saved to houses.csv")

    except (
        OSError,
        csv.Error,
        json.JSONDecodeError,
        ValueError,
        KeyError,
        TypeError
    ) as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
