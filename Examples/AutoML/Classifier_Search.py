import optuna

import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
import sklearn.svm
from sklearn.metrics import accuracy_score, log_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

def objective(trial):
    iris = sklearn.datasets.load_iris()
    data, labels = iris.data, iris.target
    
    train_data, val_data, train_labels, val_labels = sklearn.model_selection.train_test_split(data, labels, test_size=0.33, random_state=42)

    classifier_class = trial.suggest_categorical("classifier", [DecisionTreeClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier])    
    classifier = classifier_class()

    classifier.fit(train_data, train_labels)
    val_predictions = classifier.predict(val_data)
    val_acc = accuracy_score(val_labels, val_predictions)
    val_prob_predictions = classifier.predict_proba(val_data)
    val_loss = log_loss(val_labels, val_prob_predictions)

    return val_acc

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    print('Studying best trial : ',study.best_trial)
