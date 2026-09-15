 — Data Science × Logistic Regression





This is a data science and machine learning project. Its goal is to explore a Hogwarts student dataset and build a multiclass logistic regression classifier from scratch to predict each student's house.

The project covers the full machine learning workflow: statistical analysis, data visualization, feature selection, preprocessing, gradient descent, one-vs-rest classification, and prediction.
Project goals

    Reimplement the main behavior of pandas.DataFrame.describe() without using ready-made statistical functions.

    Explore the dataset using histograms, scatter plots, and a pair plot.

    Identify uninformative and redundant features.

    Implement logistic regression and gradient descent without a machine learning library.

    Train one binary classifier per Hogwarts house using a one-vs-rest strategy.

    Generate predictions in the required houses.csv format.

Exploratory data analysis
Descriptive statistics

describe.py calculates the following statistics for every numerical feature:

    Count

    Mean

    Standard deviation

    Minimum

    25th percentile

    Median

    75th percentile

    Maximum

The calculations are implemented manually instead of relying on functions such as mean, std, min, max, percentile, or describe.
Histogram

histogram.py compares the score distributions of the four houses for every course.

The analysis shows that Arithmancy and Care of Magical Creatures have very similar distributions across all houses, making them weak features for house classification.
Scatter plot

scatter_plot.py reveals that Astronomy and Defense Against the Dark Arts are almost perfectly negatively correlated. They therefore carry nearly identical information.
Pair plot

pair_plot.py displays the relationships between every pair of numerical features. Point colors represent Hogwarts houses, making it possible to identify:

    features that separate the houses;

    features with little discriminative information;

    strongly correlated and redundant features.

Selected features

The classifier uses the following courses:

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

The following features are excluded:
Feature	Reason
Arithmancy	Similar distribution across all houses
Care of Magical Creatures	Very low discriminative power
Defense Against the Dark Arts	Redundant with Astronomy
Logistic regression

For one student, the model first calculates a linear score:

$$
z = Xw + b
$$

The sigmoid function converts this score into a value between 0 and 1:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

The parameters are optimized with gradient descent:

$$
w \leftarrow w - \alpha \frac{X^T(\hat{y} - y)}{m}
$$

$$
b \leftarrow b - \alpha \frac{\sum(\hat{y} - y)}{m}
$$

where:

    $X$ is the feature matrix;

    $w$ contains one weight per feature;

    $b$ is the bias;

    $\alpha$ is the learning rate;

    $m$ is the number of students.

One-vs-rest classification

Because the dataset contains four houses, the project trains four independent binary classifiers:

Gryffindor vs all other houses
Hufflepuff vs all other houses
Ravenclaw vs all other houses
Slytherin vs all other houses

For each student, all four classifiers produce a score. The predicted house is the one with the highest score.
Data preprocessing

Missing values are replaced with the corresponding feature mean calculated from the training dataset.

Each feature is then standardized:

$$
x' = \frac{x - \mu}{\sigma}
$$

This prevents large-scale features from dominating gradient descent. The training means and standard deviations are stored with the model and reused during prediction.
Requirements

    Python 3.10+

    NumPy

    Matplotlib

Install the dependencies with:

python3 -m pip install numpy matplotlib

Usage
Display descriptive statistics

python3 describe.py datasets/dataset_train.csv

Display histograms

python3 histogram.py datasets/dataset_train.csv

Display the scatter plot

python3 scatter_plot.py datasets/dataset_train.csv

Display the pair plot

python3 pair_plot.py datasets/dataset_train.csv

Train the model

python3 logreg_train.py datasets/dataset_train.csv

Training generates:

weights.json

The file contains the selected features, preprocessing parameters, weights, and bias of each house classifier.
Generate predictions

python3 logreg_predict.py \
    datasets/dataset_test.csv \
    weights.json

Prediction generates houses.csv in the required format:

Index,Hogwarts House
0,Gryffindor
1,Hufflepuff
2,Ravenclaw
3,Slytherin

Main files
File	Purpose
describe.py	Computes descriptive statistics from scratch
histogram.py	Compares course distributions between houses
scatter_plot.py	Displays the relationship between two features
pair_plot.py	Compares all numerical features pairwise
logreg_train.py	Trains four one-vs-rest logistic regression models
logreg_predict.py	Predicts houses and generates houses.csv
weights.json	Stores trained parameters and preprocessing values
What I learned

    How descriptive statistics are calculated internally

    How to explore and interpret multidimensional data

    How to detect weak and redundant features

    Why feature normalization matters for gradient descent

    How logistic regression converts a linear score into a classification score

    How gradients update weights and bias iteratively

    How one-vs-rest extends binary classification to multiple classes

    How to persist a trained model and reproduce preprocessing at prediction time

Author

Martin Leinekugel — 42 Paris