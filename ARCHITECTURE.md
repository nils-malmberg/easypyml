# EasyPyML Architecture

## Vision

EasyPyML is a **learning-focused ML library** that teaches how algorithms work by implementing them from scratch in Python.

## Core Design Principles

### 1. Scikit-Learn Interface
Every estimator follows the same contract:
- fit(X, y) — Learn from training data. Return self.
- predict(X) — Make predictions.
- score(X, y) — Return accuracy (classification) or R² (regression).

### 2. Type Hints Everywhere
Every function has complete type annotations.

### 3. Comprehensive Testing
Tests written while coding, not after.

### 4. Immutable State with Dataclasses
Parameters stored in frozen dataclasses.

### 5. Pydantic for Validation
Strict validation of inputs.

## Module Organization

easypyml/
├── __init__.py              # Public API
├── base.py                  # BaseEstimator, BaseRegressor, BaseClassifier
├── linear_models/
│   ├── __init__.py
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   └── ridge_regression.py
├── tree/
│   ├── __init__.py
│   ├── decision_tree.py
│   └── tree_utils.py
├── ensemble/
│   ├── __init__.py
│   └── random_forest.py
├── neural_network/
│   ├── __init__.py
│   ├── layers.py
│   ├── optimizers.py
│   ├── losses.py
│   └── neural_network.py
├── preprocessing/
│   ├── __init__.py
│   ├── scaling.py
│   └── splitting.py
├── metrics/
│   ├── __init__.py
│   └── metrics.py
└── utils/
    ├── __init__.py
    ├── validation.py
    └── math_utils.py

## Development Phases

### Phase 1: Linear Models
- LinearRegression, LogisticRegression, Ridge
- Learn: gradient descent, loss functions, regularization

### Phase 2: Tree-Based
- DecisionTree, RandomForest
- Learn: information theory, ensemble methods

### Phase 3: Neural Networks
- Dense layers, activations, optimizers, MLP
- Learn: backpropagation, calculus, optimization

### Phase 4: Advanced (Optional)
- SVM, PCA, K-Means, Gradient Boosting
