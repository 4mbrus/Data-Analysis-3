# Technical Report: Startup Investment Classification & Risk Analysis

**Prepared by:** Ambrus Fodor, Katharina Burtscher\
**Date:** February 10, 2026\
**Subject:** Evaluation of Predictive Models for Capital Allocation Optimization

---

## 1. Executive Summary

We have developed and validated a machine learning framework to assist in the screening of high-potential investments. The project’s core objective was to balance the asymmetry between the risk of capital loss and the cost of missed growth opportunities.

**Key Results:**
* **Selected Model:** Random Forest Classifier
* **Optimal Decision Threshold:** 0.60
* **Financial Performance:** The model minimized total estimated business costs to **$172.5 Million** on the holdout set, outperforming Gradient Boosting and Logistic Regression baselines.
* **Risk Profile:** The model demonstrates high **Specificity (98.04%)**, effectively filtering out the vast majority of non-performing companies, while maintaining a **Precision of 76.57%** on positive investment recommendations.

---

## 2. Problem Statement & Financial Constraints

### 2.1 Objective
The goal is to predict the probability of a company being "Fast Growing" (defined as 30% ROI between 2013 and 2014). The business problem is framed as a binary classification task where:
* `1` (Positive): Fast-growing firm (Target).
* `0` (Negative): Non-performing or average firm.

### 2.2 Asymmetric Cost Matrix
Unlike academic accuracy metrics, our optimization function is driven by specific financial penalties provided by the investment committee. The cost of a "Bad Investment" (losing principal) is weighted as **2x more damaging** than a "Missed Opportunity."

For a one million dollar investment the loss function is the following:

| Prediction Error | Business Context | Cost (USD) |
| :--- | :--- | :--- |
| **False Negative (FN)** | **Missed Opportunity:** We reject a future unicorn. | **$300,000** |
| **False Positive (FP)** | **Bad Investment:** We invest in a failing company. | **$600,000** |

**Loss Function:**
$$\text{Total Cost} = (\text{FN} \times \$300,000) + (\text{FP} \times \$600,000)$$

---

## 3. Methodology & Implementation

To ensure robust deployment and prevent statistical errors, we implemented a rigorous training pipeline.

### 3.1 Preventing Data Leakage
A critical component of our architecture was the use of `Pipelines` to prevent data leakage. Preprocessing steps such as standardization (scaling) were fitted *only* on the training folds within the Cross-Validation loop, ensuring that test set statistics did not influence the training process.

**Code Snippet: Pipeline Implementation**
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# Pipeline ensures scaler is fit only on training data during CV
model_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()), 
    ('classifier', RandomForestClassifier(random_state=42))
])

# When calling fit(X_train, y_train), the scaler learns mean/std from X_train only.
```

### 3.2 Hyperparameter Tuning
We utilized GridSearchCV to optimize the Random Forest and Gradient Boosting models. We focused on the Brier Score (RMSE of probabilities) as our scoring metric to ensure the model output well-calibrated probabilities rather than just hard classifications.

Code Snippet: Grid Search Configuration
```python
from sklearn.model_selection import GridSearchCV

# Random Forest Hyperparameters
grid_params = {
    'classifier__max_features': [5, 6, 7, "sqrt"],
    'classifier__min_samples_split': [11, 16],
    'classifier__n_estimators': [500]
}

# Optimization for Probability Calibration (Brier Score)
grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=grid_params,
    cv=5,
    scoring='neg_brier_score', # Minimizing RMSE
    n_jobs=-1
)
```
### 3.3 Custom Business Cost Function
To select the final model, we did not rely solely on AUC or Accuracy. We wrote a custom loss function to evaluate the models based on the $300k/$600k cost structure.

Code Snippet: Financial Loss Calculation
```python
def calculate_business_cost(y_true, y_probs, threshold):
    # Convert probabilities to binary predictions based on threshold
    y_pred = (y_probs >= threshold).astype(int)
    
    # Calculate False Positives and False Negatives
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    # Apply Financial Costs
    cost_fn = 300_000
    cost_fp = 600_000
    
    total_loss = (fn * cost_fn) + (fp * cost_fp)
    return total_loss
```

### 4. Model Comparison & Selection
We evaluated four candidate models using 5-fold cross-validation and tested the top performers on the holdout set. While Gradient Boosting achieved a slightly higher raw AUC, the Random Forest model resulted in the lowest expected financial loss when applied to our specific cost matrix.

**Table 1: Model Performance Summary (Holdout Set)**
|Model|AUC|Classification Threshold|False Positives (Bad Inv) | False Negatives (Missed Opp) | Total Expected Loss|
|---|---|---|---|---|---|
| Logit X4 | 0.758 | 0.65 | 42 | 501 | $175,500,000 | 
| Lasso Logit | 0.761 | 0.62 | 48 | 493 | $176,700,000 |
Gradient Boosting | 0.78 1 | 0.68 | 45 | 491 |$174,300,000
|Random Forest (Selected)|0.775 | 0.60 | 41 | 493 | **$172,500,000**

**Selection Logic:** Although Gradient Boosting had a marginally better AUC, the Random Forest model's probability distribution allowed for a more efficient threshold cut-off (0.60), minimizing the expensive False Positives (41 vs 45) and resulting in a $1.8 Million cost saving compared to the next best model.

## 5. Final Model Evaluation: Random Forest
The selected Random Forest model was tested on the full holdout dataset ($N \approx 2700$).

### 5.1 Confusion Matrix Analysis
At the selected probability threshold of 0.60:

-| Actual Success (1)	| Actual Failure (0)
-|-|-
Predicted Investment (1) | 134 (TP) | 41 (FP)
Predicted Rejection (0) | 493 (FN) | 2046 (TN)

- True Positives (134): We correctly identified 134 high-growth companies.
- False Positives (41): We made bad investments in 41 companies (The "Traps").
- True Negatives (2046): We successfully avoided over 2,000 bad investments.

Statistical metrics of the classification model:
Metric | Value
-|-
Accuracy: | 80.32%  
Precision: | 76.57%  
Recall | 21.37%  
Specificity | 98.04%

5.2 Financial Impact Breakdown
Using the user-defined costs:

Cost of Missed Opportunities (FN):

$$ 493 \text{ Missed Deals} \times \$300,000 = \textbf{\$147,900,000}$$

Analysis: This is the largest component of our "cost." The model is conservative, preferring to miss a deal rather than lose capital.

Cost of Bad Investments (FP):
$$ 41 \text{ Failed Investments} \times \$600,000 = \textbf{\$24,600,000} $$

Analysis: Direct capital loss is minimized. Only ~14% of our total expected cost comes from actual cash loss; the rest is theoretical lost profit.

Total Business Cost:
$$ \$147.9M + \$24.6M = \textbf{\$172,500,000} $$

### 5.3 Sector Analysis
We performed a post-hoc analysis on two industry subsets:

- Manufacturing Sector: The model struggled, with a Recall of only 11.05%.

- Services Sector: The model performed significantly better here, achieving a Recall of 25.56%.

*Recommendation*: Use this model primarily for Service-sector candidates. Manufacturing candidates may require a lower threshold or human intervention.

## 6. Conclusion & Recommendations
The Random Forest model is a "Capital Shield." It is designed to be highly specific (98.04% Specificity), protecting the firm's principal investment at the expense of deal volume.

**Action Plan**:

- Automated Filter: Deploy the model to automatically reject applicants with a probability score below 0.60. This will safely remove ~80% of the deal flow pipeline (mostly True Negatives).

- Due Diligence: The remaining ~20% of applicants (predicted "Yes") have a Precision of 76.6%. Investment analysts should focus their deep-dive efforts here, as 3 out of 4 of these companies are likely winners.

- Sector Adjustment: Be cautious when applying this model to Manufacturing firms; consider lowering the threshold to 0.50 for that sector to catch more opportunities.