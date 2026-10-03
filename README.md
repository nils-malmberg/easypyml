# EasyPyML

**A machine learning library built from scratch in Python.**

EasyPyML is a learning-focused library that implements core ML algorithms without relying on high-level frameworks. Each algorithm is built from scratch using NumPy, following scikit-learn's interface conventions.

## Goal

Deeply understand how machine learning algorithms work by implementing them myself, with:
- Clear mathematical foundations
- Production-quality Python code
- Comprehensive tests
- Benchmarks against scikit-learn

## Features

- 📚 **Learn by Doing** — Implement algorithms from scratch
- 🎯 **Scikit-Learn Interface** — Familiar `fit()`, `predict()`, `score()` API
- 🧪 **Fully Tested** — Unit tests + benchmarks vs scikit-learn
- 📊 **Type-Hinted** — Full type hints for clarity
- 🔬 **Well-Documented** — NumPy-style docstrings
- 🚀 **Production-Ready** — CI/CD automated deployment

## Installation

```bash
uv add easypyml
```

## Quick Start

```python
import numpy as np
from easypyml.linear_models import LinearRegression

# Create sample data
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([1, 2, 3])

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict
predictions = model.predict(X)

# Score
score = model.score(X, y)
print(f"R² Score: {score}")
```

## Algorithms Implemented

### Phase 1: Linear Models
- LinearRegression — Simple linear regression (closed-form solution)
- LogisticRegression — Binary classification with gradient descent
- RidgeRegression — Linear regression with L2 regularization

### Phase 2: Tree-Based Models
- DecisionTree — Classification and regression trees
- RandomForest — Ensemble of decision trees

### Phase 3: Neural Networks
- MLPRegressor — Multi-layer perceptron for regression
- MLPClassifier — Multi-layer perceptron for classification

### Phase 4: Advanced (Optional)
- SVM, PCA, K-Means, Gradient Boosting

## Documentation

- [Architecture & Design](./ARCHITECTURE.md) — Deep dive into structure and patterns
- [Contributing & Pair Programming](./CONTRIBUTING.md) — How to develop with Claude Code
- [Getting Started](./GETTING_STARTED.md) — Tutorial and examples
- [Setup Instructions](./REPO-SETUP.md) — GitHub + local setup

## Status

🚧 **Early Development** — Phase 1 (Linear Models) in progress

## License

MIT

## Pair Programming

This project is developed through pair programming with Claude Code.
