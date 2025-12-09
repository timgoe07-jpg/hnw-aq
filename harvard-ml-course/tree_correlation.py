#! /usr/bin/env python3
# tree_correlation.py

import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
import dtreeviz


def save_tree(viz, filename, **view_kwargs):
    """
    Call viz.view(...) to get a render object, then save to SVG.
    """
    render = viz.view(**view_kwargs)  # this returns an object with .save()
    filepath = os.path.abspath(filename)
    render.save(filepath)
    print(f"[Saved] {filepath}")


def main():
    # Read data
    df = pd.read_csv("diabetes.csv")

    # Predictors and target
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    random_state = 144

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, train_size=0.8, random_state=random_state
    )

    # -----------------------------
    # Bagging
    # -----------------------------
    max_depth = 20
    n_estimators = 1000

    basemodel = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state
    )

    bagging = BaggingClassifier(
        estimator=basemodel,
        n_estimators=n_estimators,
        random_state=random_state
    )
    bagging.fit(X_train, y_train)

    bag_pred = bagging.predict(X_val)
    acc_bag = round(accuracy_score(y_val, bag_pred), 2)
    print(f"For Bagging, the accuracy on the validation set is {acc_bag}")

    # -----------------------------
    # Random Forest
    # -----------------------------
    rf = RandomForestClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        random_state=random_state
    )
    rf.fit(X_train, y_train)

    rf_pred = rf.predict(X_val)
    acc_rf = round(accuracy_score(y_val, rf_pred), 2)
    print(f"For Random Forest, the accuracy on the validation set is {acc_rf}")

    # -----------------------------
    # Visualizations
    # -----------------------------
    # Use shallower trees to make diagrams readable
    max_depth_vis = 3

    basemodel_vis = DecisionTreeClassifier(
        max_depth=max_depth_vis,
        random_state=random_state
    )

    bagging_vis = BaggingClassifier(
        estimator=basemodel_vis,
        n_estimators=1000,
        random_state=random_state
    )
    bagging_vis.fit(X_train, y_train)

    # Grab two trees from Bagging
    bagvati1 = bagging_vis.estimators_[0]
    bagvati2 = bagging_vis.estimators_[100]

    X_full = df.drop("Outcome", axis=1)
    y_full = df["Outcome"]

    viz_common_kwargs = dict(
        X_train=X_full,
        y_train=y_full,
        feature_names=X_full.columns,
        target_name="Diabetes",
        class_names=["No", "Yes"],
    )

    # ---- Bagging Tree 1 ----
    vizA = dtreeviz.model(
        bagvati1,
        **viz_common_kwargs
    )
    save_tree(vizA, "bagging_tree_1.svg", scale=1.4)

    # ---- Bagging Tree 2 ----
    vizB = dtreeviz.model(
        bagvati2,
        **viz_common_kwargs
    )
    save_tree(vizB, "bagging_tree_2.svg", scale=1.4)

    # ---- Random Forest visualizations ----
    rf_vis = RandomForestClassifier(
        max_depth=max_depth_vis,
        n_estimators=1000,
        random_state=random_state,
        max_features="sqrt",
    )
    rf_vis.fit(X_train, y_train)

    forest1 = rf_vis.estimators_[0]
    forest2 = rf_vis.estimators_[100]

    vizC = dtreeviz.model(
        forest1,
        **viz_common_kwargs
    )
    save_tree(vizC, "rf_tree_1.svg", scale=1.4)

    vizD = dtreeviz.model(
        forest2,
        **viz_common_kwargs
    )
    save_tree(vizD, "rf_tree_2.svg", scale=1.4)


if __name__ == "__main__":
    main()
