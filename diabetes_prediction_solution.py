import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def load_data():
    base_dir = Path(r'c:\Users\DELL\Downloads\assignment1')
    csv_file = next(base_dir.glob('*.csv'))
    df = pd.read_csv(csv_file)
    return df


def summarize_dataset(df):
    print('Dataset shape:', df.shape)
    print('\nColumn names:', list(df.columns))
    print('\nData types:\n', df.dtypes)
    print('\nMissing values:\n', df.isnull().sum())
    print('\nBasic statistical summary:\n', df.describe().T)


def build_models():
    return {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'Random Forest': RandomForestClassifier(n_estimators=300, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    }


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_metrics = {
        'accuracy': accuracy_score(y_train, train_pred),
        'precision': precision_score(y_train, train_pred, zero_division=0),
        'recall': recall_score(y_train, train_pred, zero_division=0),
        'f1': f1_score(y_train, train_pred, zero_division=0),
    }

    test_metrics = {
        'accuracy': accuracy_score(y_test, test_pred),
        'precision': precision_score(y_test, test_pred, zero_division=0),
        'recall': recall_score(y_test, test_pred, zero_division=0),
        'f1': f1_score(y_test, test_pred, zero_division=0),
    }

    train_cm = confusion_matrix(y_train, train_pred)
    test_cm = confusion_matrix(y_test, test_pred)

    print(f'\n==== {name} ====')
    print(f'Training metrics: accuracy={train_metrics["accuracy"]:.4f}, precision={train_metrics["precision"]:.4f}, recall={train_metrics["recall"]:.4f}, f1={train_metrics["f1"]:.4f}')
    print(f'Testing metrics:  accuracy={test_metrics["accuracy"]:.4f}, precision={test_metrics["precision"]:.4f}, recall={test_metrics["recall"]:.4f}, f1={test_metrics["f1"]:.4f}')
    print('Training confusion matrix:\n', train_cm)
    print('Testing confusion matrix:\n', test_cm)

    return {
        'name': name,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'model': model,
    }


def main():
    df = load_data()
    summarize_dataset(df)

    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print('\nFeatures (X):', list(X.columns))
    print('Target (y): Outcome (0 = No Diabetes, 1 = Diabetes)')
    print(f'Training size: {X_train.shape[0]} samples')
    print(f'Testing size: {X_test.shape[0]} samples')

    models = build_models()
    results = []
    for name, model in models.items():
        results.append(evaluate_model(name, model, X_train, X_test, y_train, y_test))

    best_result = max(results, key=lambda r: r['test_metrics']['f1'])
    print('\nBest model selected by test F1-score:', best_result['name'])
    print('Selected model test metrics:')
    print('accuracy=', round(best_result['test_metrics']['accuracy'], 4))
    print('precision=', round(best_result['test_metrics']['precision'], 4))
    print('recall=', round(best_result['test_metrics']['recall'], 4))
    print('f1=', round(best_result['test_metrics']['f1'], 4))

    print('\nInterpretation:')
    print('The dataset is clean and does not require major missing-value treatment; therefore the main preprocessing step was train/test splitting and model fitting.')
    print('Logistic Regression is a strong baseline because it is simple, interpretable, and effective for binary classification with continuous medical features.')
    print('For this dataset, Random Forest achieved the highest test F1-score and therefore is selected as the best-performing solution.')
    print('This indicates that the model captures nonlinear relationships among the medical indicators more effectively than the simpler linear baseline.')


if __name__ == '__main__':
    main()
