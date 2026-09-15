import csv
import sys
import json

import numpy as np

from describe import (
    ft_mean,
    ft_std,
    get_numeric_values,
)


HOUSES = [
    "Gryffindor",
    "Hufflepuff",
    "Ravenclaw",
    "Slytherin",
]

FEATURES = [
    "Astronomy",
    "Herbology",
    "Divination",
    "Muggle Studies",
    "Ancient Runes",
    "History of Magic",
    "Transfiguration",
    "Potions",
    "Charms",
    "Flying",
]


def load_dataset(
    filename: str
) -> list[dict[str, str]]:
    with open(filename, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader)


def get_feature_mean(
    dataset: list[dict[str, str]],
    feature: str
) -> float:
    values = get_numeric_values(
        dataset,
        feature
    )

    if not values:
        return 0.0

    return ft_mean(values)

# creer features, et liste de moyennes
def prepare_features(
    dataset: list[dict[str, str]]
) -> tuple[np.ndarray, list[float]]: # matrice numpy (features) avec notes, et moyenne notes (means) dans un tuple
    means = []

    # calcul moyennes
    for feature in FEATURES:
        mean = get_feature_mean(
            dataset,
            feature
        )
        means.append(mean)

    rows = [] # matrice avec une ligne par eleve

    for student in dataset:
        row = [] #on construit une ligne

        for index, feature in enumerate(FEATURES):
            value = student[feature]

            if value == "":
                row.append(means[index]) # si valeur vide, on met la moyenne
            else:
                row.append(float(value))

        rows.append(row) #on ajoute la ligne construite a la matrice

    return np.array(rows), means


def prepare_labels( # recup maison de chaque eleve
    dataset: list[dict[str, str]]
) -> list[str]:
    labels = []

    for student in dataset:
        labels.append(
            student["Hogwarts House"]
        )

    return labels



# features.shape == (1600, 10), 1600 lignes de 10 collones
# features[0] correspond a un eleve (ligne)
# features[:, 0] correspond a une matiere (colonne)
def normalize_features(
    features: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = []
    standard_deviations = [] # ecart type matieres

    for index in range(features.shape[1]): # (shape[1] = nbre de colonne = nbre de matieres = 10) -> boucle sur les matieres (index)
        column = features[:, index].tolist() # creer une liste de notes par matieres -> boucle sur les eleves pour un index

        means.append(
            ft_mean(column) # moyenne d'une matiere (index) sur tous les eleves (de toutes les maisons)
        )
        standard_deviations.append(
            ft_std(column) # ecart type par matiere
        )

    means = np.array(means) #conversion en tableau numpy pour additioner, soustraire

    standard_deviations = np.array(
        standard_deviations
    )

    standard_deviations[
        standard_deviations == 0
    ] = 1.0 # si ecart type = 0, on met 1 pour eviter division par 0

    normalized_features = ( # normaliser : Xnorm = (x - mean) / ecart-type
        features - means # pour chaque ligne de features (notes), on lui soustrait la moyenne de la matiere correspondante (situer au meme index)
    ) / standard_deviations

    return (
        normalized_features,
        means,
        standard_deviations
    )

def create_binary_labels( # creer une liste de 1600 1.0 ou 0.0 pour chaque maison
    labels: list[str],
    target_house: str
) -> np.ndarray:
    binary_labels = []

    for label in labels:
        if label == target_house:
            binary_labels.append(1.0)
        else:
            binary_labels.append(0.0)

    return np.array(binary_labels)

# transforme valeur en -> x appartenant a [0,1]
def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(
        values,
        -500, # valeur min et max pour l'exp
        500
    )

    return 1.0 / (
        1.0 + np.exp(-values) # 1 / (1 + exp(-x)) pour chaque values
    ) # si x -> -500 ; f(x) -> 0  
      # si x-> 0 ; f(x) -> 0.5
      # si x -> 500 ; f(x) -> 1
        # x petit -> 0; x grand -> 1

# on fait cette fonction pour chaque maison donc 4 fois, avec son propre labels (boucle for dans le main)
def train_binary_classifier(
    features: np.ndarray,
    labels: np.ndarray, # change pour chaque maison, 1600 lignes (1.0 si eleve appartient a la maison, 0.0 sinon)
    learning_rate: float,
    iterations: int
) -> tuple[np.ndarray, float]: #weights, bias
    student_count = features.shape[0] # 1600
    feature_count = features.shape[1] # 10

    weights = np.zeros(feature_count) # tableau de 0.0 de taille 10 (1 poid par matiere)
    # plus le poids d'une matiere est grand, plus elle est importante pour la maison

    bias = 0.0 # ajustement general

    for _ in range(iterations): # on fait la boucle pour la proba d'une seul maison, on fera 4 fois la fction
        linear_results = (
            features @ weights # @ = multiplication matricielle   (ici features = 1600x10, weights = 10x1) donne 1600x1
        ) + bias
        # premiere iteration : que des 0

        predictions = sigmoid(
            linear_results # met en proba, 0 -> 0,5 (1 / 1 + e^0)
        )  
        # premiere iteration : que des 0.5

        errors = predictions - labels # 0,5 -> - 0,5 si bonne maison (manque 0,5), -> 0,5 sinon (0,5 de trop)
        # errors est un tableau de 1600 valeurs qui evolue par comparaison avec labels

        weight_gradient = (
            errors @ features # fait la somme pour chaque matiere de toutes ses notes * erreur eleve
                              # a chaque iteration : note eleve 1 en astro * erreur1 + note eleve 2 en astro * erreur2 + ... + note eleve 1600 en astro * erreur1600
                              # pareil pour toutes les matieres -> donne un tableau de 10 valeurs (1 par matiere)
                              # si bonne note dans une matiere et bonne prediction -> va augmenter le poids de cette matiere,
                              # si mauvaise note et bonne prediction -> va diminuer le poids de cette matiere
        ) / student_count # donne la moyenne de l'erreur par matiere (somme contribution de chaque eleve / nbre d'eleves) pour que la taille du dataset influence pas
        
        bias_gradient = ft_mean( # 1 seul valeur, moyenne d'erreur
            errors.tolist() 
        )

        weights -= (
            learning_rate
            * weight_gradient
        )

        bias -= ( # si bias_gradient > 0, predictions trop elevees, donc on fait -=,
                  # pareil si bias_gradient < 0, predictions trop basses, on fait -= aussi pour augmenter bias 
            learning_rate
            * bias_gradient
        )

    return weights, bias

def save_model(
    filename: str,
    models: dict,
    missing_value_means: list[float],
    normalization_means: np.ndarray,
    normalization_stds: np.ndarray
) -> None:
    data = {
        "features": FEATURES,
        "houses": HOUSES,
        "missing_value_means": (
            missing_value_means
        ),
        "normalization_means": (
            normalization_means.tolist()
        ),
        "normalization_stds": (
            normalization_stds.tolist()
        ),
        "models": {},
    }

    for house in HOUSES:
        data["models"][house] = {
            "weights": (
                models[house]["weights"].tolist()
            ),
            "bias": float(models[house]["bias"]),
        }

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            data,
            file,
            indent=4
        )




def main() -> None:
    if len(sys.argv) != 2:
        print(
            "Usage: python3 logreg_train.py "
            "<dataset_train.csv>"
        )
        return

    try:
        dataset = load_dataset(sys.argv[1])

        if not dataset:
            print("Error: dataset is empty")
            return

        features, missing_value_means = (
            prepare_features(dataset)
        ) # features = tableau de tableau representant les notes de chaque eleves

        labels = prepare_labels(dataset) # liste des maisons de chaque eleve (1600 eleves)

        (
            normalized_features,
            normalization_means,
            normalization_stds,
        ) = normalize_features(features)

        models = {}

        for house in HOUSES:
            print(f"Training {house}...")

            binary_labels = create_binary_labels(
                labels,
                house
            )

            weights, bias = train_binary_classifier(
                normalized_features,
                binary_labels,
                learning_rate=0.1,
                iterations=10000
            )

            models[house] = {
                "weights": weights,
                "bias": bias,
            }

        save_model(
        "weights.json",
        models,
        missing_value_means,
        normalization_means,
        normalization_stds
        )

        print("Model saved to weights.json")
            
    except (
        OSError,
        csv.Error,
        ValueError,
        KeyError
    ) as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()