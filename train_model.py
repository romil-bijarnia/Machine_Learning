import sys

try:
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

except ModuleNotFoundError as e:
    missing = str(e).split("'")[1]
    sys.exit(
        f"Missing dependency: {missing}.\n"
        "Install it with `pip install scikit-learn` and retry."
    )

except ModuleNotFoundError:
    print(
        "This example requires scikit-learn. "
        "Install it with 'pip install scikit-learn' to run the demo."
    )
    sys.exit(0)



def main():
    # Load iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create and train logistic regression model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.2f}")

    # Perform simple cross validation for a more robust score
    from sklearn.model_selection import cross_val_score

    scores = cross_val_score(model, X, y, cv=5)
    print(
        f"5-fold cross validation accuracy: {scores.mean():.2f} \u00b1 {scores.std():.2f}"
    )


if __name__ == "__main__":
    main()
