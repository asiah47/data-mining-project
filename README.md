# 📊 Data Mining Assignment

This repository contains my completed solution for the Data Mining module assignment. The assignment is split into two major parts:

- **Part 1**: Implementation of a custom Nearest Neighbour classifier using the Minkowski distance (no libraries).
- **Part 2**: Development and evaluation of machine learning models to predict credit card default, using multiple classification algorithms.

---

##  Files Included

### Part 1 (Nearest Neighbour Classifier)
| File | Description |
|------|-------------|
| `sonar_train.csv` | Training dataset for sonar classification |
| `sonar_test.csv` | Test dataset for sonar classification |
| `Part1_Asia.ipynb` | Jupyter notebook implementing Part 1(a), evaluation metrics, and results |

### Part 2 (Credit Risk Prediction)
| File | Description |
|------|-------------|
| `creditdefault_train.csv` | Training dataset for credit default prediction |
| `creditdefault_test.csv` | Test dataset for credit default prediction |
| `Part2_Asia.ipynb` | Jupyter notebook implementing and evaluating six ML models with tuning, charts, and conclusion |

---

## Assignment Tasks Summary

### **Part 1 – Nearest Neighbour (from scratch)**
- Used the **Sonar dataset** to classify objects as Rock or Metal (`R` or `M`) based on 60 numerical features.
- Implemented the **simple Nearest Neighbour algorithm** using **Minkowski distance**.
- Calculated **accuracy**, **precision**, **recall**, and **F1-score** for class `M`.
- Compared performance for various `q` values (from 1 to 20) in the Minkowski formula.
- Visualized the effect of `q` on performance metrics.

### **Part 2 – Credit Risk Modeling**
- Used real-world credit card payment data with 23 predictors.
- Built and tuned classification models using:
  - k-Nearest Neighbours
  - Decision Trees
  - Random Forest
  - Bagging
  - XGBoost (in place of AdaBoost)
  - Support Vector Machines (SVM)
- Applied **cross-validation**, **hyperparameter tuning**, and model comparison.
- Selected the **best-performing model** and justified selection with relevant metrics.
- Included charts showing how accuracy varies with one hyperparameter for each model.


---

## 🧪 Evaluation Metrics Used
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Cross-Validation Scores
- Hyperparameter Tuning Results

---

## 📊 Visualisations
- Line plots of performance vs. Minkowski distance power (`q`)
- Accuracy vs. hyperparameter plots (e.g., `k`, `max_depth`, `C`, etc.)
- Confusion matrices for test predictions

---

## 📖 Notes

- All implementations are done in **Jupyter Notebooks** using Python 3.
- Part 1 is coded from scratch without using ML libraries for distance or classification.
- Part 2 makes use of `scikit-learn`, `xgboost`, `matplotlib`, and `pandas`.
- This work was completed independently and submitted as part of university coursework.

---

## Technologies & Libraries

- Python 3.x
- Jupyter Notebook
- NumPy
- pandas
- scikit-learn
- matplotlib / seaborn
- xgboost

---

##  What I Learned

- How distance metrics influence classification performance
- Implementing nearest neighbour classifiers manually
- Model evaluation and hyperparameter tuning
- Comparative analysis across machine learning algorithms
- Practical data preprocessing and metric-based decision making

---

## License

This project is submitted as part of academic coursework and is intended for educational purposes only.
