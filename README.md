 — Data Science × Logistic Regression


DSLR is a machine learning project from the 42 curriculum. The goal is to predict a Hogwarts student's house by implementing multiclass logistic regression from scratch, without using a machine learning library.

The project focuses on feature preprocessing, logistic regression, gradient descent, and one-vs-rest classification.

Model pipeline

dataset_train.csv
        ↓
Missing-value imputation
        ↓
Feature standardization
        ↓
Four one-vs-rest classifiers
        ↓
Gradient descent
        ↓
weights.json
        ↓
Predictions on dataset_test.csv
        ↓
houses.csv

Selected features

Exploratory analysis is used to remove uninformative or redundant courses. The model is trained with:

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

Arithmancy and Care of Magical Creatures are excluded because their distributions are similar across all houses. Defense Against the Dark Arts is excluded because it is redundant with Astronomy.

Preprocessing

Missing values are replaced with the corresponding feature mean calculated from the training dataset.

Because course scores use very different scales, every feature is standardized:

$$
x' = \frac{x - \mu}{\sigma}
$$

After standardization, the features are centered around zero and use comparable scales. This prevents large-valued courses from dominating the gradients.

The training means and standard deviations are saved with the model and reused unchanged during prediction.

Logistic regression

For one student, a binary classifier first computes a linear score:

$$
z = Xw + b
$$

The sigmoid function transforms this score into a value between 0 and 1:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

A large positive score produces a result close to 1, while a large negative score produces a result close to 0.

Gradient descent

The weights start at zero and are improved iteratively. At each iteration, logreg_train.py performs:

linear_results = features @ weights + bias
predictions = sigmoid(linear_results)
errors = predictions - labels

weight_gradient = (
    errors @ features
) / student_count

bias_gradient = mean(errors)

weights -= learning_rate * weight_gradient
bias -= learning_rate * bias_gradient

The gradients are:

$$
\nabla_w J = \frac{1}{m}X^T(\hat{y} - y)
$$

$$
\frac{\partial J}{\partial b}
= \frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i-y_i)
$$

The parameters are updated in the opposite direction of the gradient:

$$
w \leftarrow w - \alpha \nabla_w J
$$

$$
b \leftarrow b - \alpha \frac{\partial J}{\partial b}
$$

Here, $m$ is the number of students and $\alpha$ is the learning rate.

One-vs-rest classification

Logistic regression is binary, while Hogwarts has four houses. The training program therefore builds four classifiers:

Gryffindor vs all
Hufflepuff vs all
Ravenclaw vs all
Slytherin vs all

For each classifier, target labels are converted to 1 for the current house and 0 for every other house.

During prediction, the four models calculate one score each. The house with the highest score is selected.

Usage

Install the dependencies:

python3 -m pip install numpy matplotlib

Train the four classifiers:

python3 logreg_train.py datasets/dataset_train.csv

This generates weights.json, containing the preprocessing values, weights, and bias of every classifier.

Generate predictions:

python3 logreg_predict.py \
    datasets/dataset_test.csv \
    weights.json

The output is written to houses.csv:

Index,Hogwarts House
0,Gryffindor
1,Hufflepuff
2,Ravenclaw

Main files

File

Purpose

logreg_train.py

Preprocesses the data and trains four classifiers with gradient descent

logreg_predict.py

Loads the model and predicts each student's house

weights.json

Stores preprocessing parameters, weights, and biases

houses.csv

Contains the final predictions

Key concepts

Feature imputation and standardization

Vectorized matrix operations

Sigmoid activation

Binary cross-entropy gradient

Gradient descent

One-vs-rest multiclass classification

Model serialization and reproducible preprocessing

Author

Martin Leinekugel — 42 Paris