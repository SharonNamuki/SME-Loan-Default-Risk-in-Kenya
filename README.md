** Predicting SME Loan Default Risk in Kenya Using Gradient Boosting & Explainable AI**
1. **Project Overview**
   
Small and Medium Enterprises (SMEs) and youth entrepreneurs in emerging markets often face limited access to affordable credit due to high perceived default risk. Traditional credit assessment methods are manual, slow, subjective, and difficult to scale.
This project develops a supervised machine learning framework for early loan default risk detection, enabling lenders to:

    Reduce non-performing loans (NPLs)
    Identify high-risk borrowers early
    Enable proactive intervention strategies
    Promote financial inclusion through fairer, data-driven credit decisions

2.  **Data Understanding**
   
 2.1 **Data Source**
 
The dataset consists of loan-level data collected from Micromart Africa’s digital lending platform, focusing on SME and youth entrepreneur borrowers.

 2.2 **Data Overview**

Total records: 277,159 loans

Problem type: Binary classification

Target variable: is_default (1 = Default, 0 = Non-default)

2.3 **Feature Categories**

Borrower Attributes: Age, Gender, Account Age

Loan Characteristics: Principal Amount, Interest, Loan Term, Product Type, Number of Installments

Repayment Behavior: Total paid, Unpaid Balance, Days Late, Number of Late Payments, Number of on-time payments

The dataset combines demographic, transactional, and behavioral data, making it suitable for supervised credit risk modeling.


3. **Data Cleaning & Preprocessing**
   
3.1 **Missing Values**

Extreme cases with high levels of missing data were dropped to reduce noise and improve model reliability.
We dropped unique IDs used for tracking and do not provide predictive value for loan repayment behavior

3.2 **Duplicates**
Our data had no duplicate values


3.3 **Exploratory Data Analysis(EDA)**

3.3.1 **Univariate analysis**

This was conducted to understand feature distributions.

Target Variable: is_default

<img width="860" height="354" alt="Univariate" src="https://github.com/user-attachments/assets/55fe6d10-7f36-40e9-9427-583bf645b9a5" />

**Interpretation**

-The dataset exhibited moderate class Imbalance so our Evaluation matrix included

        ◦ Precision & Recall (especially for the minority "Default" class)
        ◦ F1-Score (balance between precision and recall)
        ◦ ROC-AUC (overall discrimination ability)
        ◦ Confusion Matrix (visualize trade-offs)

**Recommendation**

-For Data splitting, we used stratified sampling (train/test/validation) to maintain the 80:20 class ratio.

Model Selection-Algorithms that handle model imbalance well:
        ◦ Random Forest (class weighting)
        ◦ XGBoost (scale_pos_weight parameter)
        ◦ Logistic Regression (class_weight='balanced')


3.3.2 **Bivariate analysis**

This was used to assess relationships between features and loan default risk.

<img width="878" height="394" alt="Bivariate" src="https://github.com/user-attachments/assets/5616195c-2c2d-4cee-8a1e-050866568939" />

**Findings**

1. Default Rate by Quarter (Seasonal Pattern)
         • Q1: 19.23%
         • Q2: 20.17%
         • Q3: 20.73% (peak)
         • Q4: 18.85% (lowest)
This implies that Q3 carries the highest risk, possibly due to seasonal economic factors or lending cycles.

2. Default Rate by Gender (Minimal Difference)
    • Gender 1.0: 19.51% default
    • Gender 2.0: 19.88% default
    • The difference is negligible (<0.4 percentage points), indicating gender is not a strong differentiator of          default risk.
    
3. Default Rate by Loan Type (Significant Difference)
    • New Loans: 25.29% default rate
    • Existing Loans: 17.99% default rate
    • Difference: 7.3 percentage points
    
New loans are significantly riskier, which aligns with typical lending patterns, even in the current financial industry, where new borrowers have unproven creditworthiness.

4. Top 10 Products by Default Rate
    • Product 30149 has 100% default rate but only 2 loans (small sample).
    • Other high-risk products have default rates ranging from 40% to 83%.
    • These high-risk products represent specific segments that may need product-level risk review.

**Payment Behaviour Analysis**

<img width="870" height="502" alt="PBA" src="https://github.com/user-attachments/assets/d4a29aaf-aec7-4e3d-b96b-00e9b02ed155" />

**Interpretations and Recommendations**

1. Default is Preceeded by Clear Delinquency Patterns
Defaulters show classic delinquency patterns:
    • Chronic lateness (avg 222 days late)
    • Maximum lateness extending over a year
    • Penalty accumulation
    • Reduced payment amounts and frequency
    
2. Actionable Early Warning Signals
Monitor these thresholds for early intervention:
    • avg_days_late > 30 days
    • max_days_late > 90 days
    • Any penalties incurred
    • num_ontime_payments below 3
    • avg_installment_paid dropping >20%
    
Recommendations:
    1. Key features for default prediction models are
        max_days_late, avg_days_late, num_ontime_payments, penalty_rate
        
    2. Create a composite "delinquency score" combining:
        ◦ Days late metrics
        ◦ Penalty indicators
        ◦ Payment frequency
        
    3. Segment customers by payment behavior:
        ◦ "At-risk": High days late + penalties
        ◦ "Watchlist": Reduced payment amounts/frequency
        ◦ "Low-risk": Consistent on-time payments, no penalties

**Loan Characteristics Analysis**

<img width="1" height="1" alt="Loan Characteristics distribution" src="https://github.com/user-attachments/assets/3ed984a6-2618-430b-9d51-bd0723c812fb" />

Interpretation and Recommendation

1. Loan Amount vs. Default
         ◦ Small loan amounts have more defaulters
         ◦ Large loan amounts have fewer defaulters
   Implying that small loans carry disproportionate risk.

2. Interest Rate vs. Default-both very high and very low interest rates are associated with default

3. Loan Term Days vs. Default
4. 
Defaulters tend to have shorter loan terms compared to non-defaulters. This is because they require larger periodic payments, less time for financial recovery and may divert the purpose

 **Customer Demographics Analysis**

   1. Age
      
<img width="308" height="313" alt="Default Rate by age" src="https://github.com/user-attachments/assets/0823040a-2f3d-4c52-b90e-ec27c03a3536" />

**Findings**

More default is experienced in younger years compared to older years

2. Gender

<img width="884" height="382" alt="Gender Analysis" src="https://github.com/user-attachments/assets/046c7104-76d0-434f-b78d-bcd86d15a806" />

Findings- No significant difference in default across both genders

3. Account Age Risk Analysis

4.<img width="316" height="280" alt="Account Age Distribution" src="https://github.com/user-attachments/assets/07f93f03-5ce8-4c7b-ad6b-249f1ea5041b" />
 
Findings- Default risk decreases with  account age

3.3.3 **Multivariate Analysis**

**Correlation Matrix Heatmap(Numerical Variables)**
<img width="578" height="495" alt="Correlation matrix" src="https://github.com/user-attachments/assets/9495acaa-8922-4296-8fe4-3a438e356cbb" />

1. Strongest Correlations with Default (is_default)
    • max_days_late: 0.65 (strong positive)
    • avg_days_late: 0.58 (strong positive)
    • previous_loan_count: -0.10 (weak negative)
    • age: -0.07 (weak negative)
    • account_age_days: -0.07 (weak negative)
This implies that higher delinquency (more days late) strongly predicts default, while more experienced borrowers (older, more loans) are slightly less likely to default.

2. High Feature Collinearity (Potential Redundancy)
    • max_days_late & avg_days_late: 0.91 (very strong correlation)
    • account_age_days & previous_loan_count: 0.83 (strong correlation)
    • loan amount & previous_loan_count: 0.45 (moderate correlation)
    • loan amount & interest_rate: 0.35 (moderate correlation)
    
3. Weak or Negligible Correlations with Default
    • total_unpaid_amount: 0.001 (almost no correlation)
    • loan_to_limit_ratio: -0.01 (almost no correlation)
    • num_late_payments: -0.04 (very weak negative)
    
4. Notable Behavioral Feature Relationships
    • late_payment_rate & num_late_payments: 0.69 (moderately strong)
    • interest_rate & num_late_payments: 0.19 (weak positive)
    • repayment_ratio & loanamount: 0.22 (weak positive)

   
**Baseline Model**

For this use case:
    • Recall is the most critical metric → We want to get as many potential defaults as possible (minimize false negatives)
    • Precision is important but secondary → Some false positives are acceptable if we catch more true defaults

   **Test Set Results (Most Important)**
   
    • Recall: 92.53% → Captures 92.5% of actual defaults (good for           early warning)
    • Precision: 97.64% → When the model predicts "default", it's            correct 97.6% of the time
    • F1-Score: 95.01% → Strong balance between precision and recall
    • Accuracy: 98.07% → Overall prediction accuracy
    
**Key Takeaways for Business**

    1. High default detection rate: 92.5% recall means the model successfully identifies most at-risk loans
    2. Low false alarm rate: 97.6% precision means only ~2.4% of default warnings are false alarms
    3. Model is reliable: Test performance closely matches training performance
    4. Good starting point: This baseline model provides a solid foundation for early default detection

Payment behavior is the strongest predictor of loan default
Certain loan products (e.g., Product IDs 30150, 30154, 30111) exhibit higher risk
Seasonality effects were observed, with December loans showing elevated default risk

**Feature Importance**
**Top 15 Most Important Features**

<img width="574" height="404" alt="Top 15 Most Important Features" src="https://github.com/user-attachments/assets/024ffade-9112-4372-a4ca-972ae7beafac" />

**Advanced Models**

Models Choosen

🔹 Random Forest

It captures nonlinear interactions, Robust to noisy features and Strong with behavioral data

Random Forest Model Performance

<img width="362" height="65" alt="Random Forest Performance" src="https://github.com/user-attachments/assets/d47b8c1c-b148-41a4-b437-0592dd9e4cde" />

**Random Forest for Early Loan Default Risk Detection**

Better Default Detection (Recall)

    • Random Forest catches 93.95% of defaults vs. 92.53% for Logistic Regression
    • Additional 1.42% of at-risk loans identified → potentially significant business impact
    • Closer to the Best CV Recall of 98.12% during hyperparameter tuning
    
Higher Precision

    • 98.83% precision means even fewer false alarms
    • When RF predicts "default", it's correct 98.8% of the time
    • Reduces operational costs of investigating false positives
    
Potential Overfitting Concern

    • Training metrics near perfect (99.99%) but test metrics are lower
    • This gap suggests some overfitting, but still strong generalization
    • Business implication: Model is highly optimized on historical data
    
**Business Implications for Early Default Detection:**

Advantages of Random Forest:

    1. Higher recall → Catches more potential defaults (critical for risk management)
    
    2. Higher precision → Fewer false alarms (reduces operational costs)
    
    3. Faster training → More efficient model development
    
    4. Non-linear relationships → Better captures complex patterns in loan data
    
**Considerations:**

    1. Slight overfitting → Monitor performance on new data closely
    
    2. Model complexity → Harder to interpret than logistic regression

Top 15 Most Important Features(Random Forest)
<img width="586" height="415" alt="Top 15 most important-random forest" src="https://github.com/user-attachments/assets/0645dd99-e69b-499e-a681-2d6f309b19ae" />


🔹 LightGBM
It is employed to model complex non-linear interactions in repayment behavior

Why LightGBM is Superior:
    1. Highest Default Detection Rate (95.84% recall)
        ◦ Catches 240 more defaults per 10,000 loans compared to Random Forest
        ◦ Critical for risk minimization in early detection systems
    2. Best Overall Performance Balance
        ◦ Highest F1-score (97.25%) → optimal precision-recall tradeoff
        ◦ Highest accuracy (98.92%) → most reliable overall predictions
    3. Computational Efficiency
        ◦ Fastest training → enables rapid model iteration
        ◦ Efficient memory usage → scalable to larger datasets
    4. Modern Algorithm Advantages
        ◦ Gradient boosting handles complex patterns better
        ◦ Built-in regularization reduces overfitting risk
        ◦ Native support for categorical features

   Business Justification for Choosing LightGBM:
1. Exceptional and Consistent Performance
    • F1-Score > 99% across all folds
    • Minimal variation (±0.07%) → model is not overfitting
    • Reliable across different data segments
2. Outstanding Default Detection (Recall)
    • CV Recall: 98.70% → catches nearly all defaults
    • Even higher than single split recall (95.84%)
    • Business impact: Maximum risk coverage with minimal missed defaults
3. Near-Perfect Precision
    • 99.64% precision → minimal false alarms
    • When model flags a loan as risky, it's correct 99.6% of the time
    • Operational efficiency: Low investigation cost for false positives
4. Model Robustness Validated
    • Low standard deviations across all metrics
    • Consistent performance across 5 different test sets
    • No data leakage or split bias concerns

**Business Interpretation**

The model correctly identifies over 92% of defaulters
Very low false alarm rate (2.4%)
Minimal overfitting observed
Strong and reliable baseline for early risk detection

6. **Advanced Models**
   
Several advanced models were evaluated, and the best-performing model was LightGBM

**LightGBM was best because......**
   It gave the Highest Recall: 95.84%
   Detects 240 more defaulters per 10,000 loans compared to Random Forest
   Best F1-score: 97.25%
   Highest Accuracy: 98.92%
   Fast training & memory-efficient
   Handles complex non-linear patterns
   Built-in regularization reduces overfitting risk
   Native support for categorical features

Conclusion: LightGBM provides the best balance between risk detection, reliability, and scalability.

**EXPORT LIGHTGBM MODEL FOR APP DEPLOYEMENT USE**

**RECOMMENDATIONS AND CONCLUSION**

1. Adoption of Machine Learning-Based Credit Scoring Financial institutions should adopt machine learning-based credit scoring systems instead of relying solely on traditional rule-based or manual credit assessments. The models developed in this study demonstrate that borrower characteristics such as num_ontime_payments, num_late_payments, avg_installment_paid, and late_payment_rate are important predictors of default. Using these, models can improve the early detection of loan defaulters and help the business make loan policies that will reduce non-performing loans and promote financial inclusion through fairer credit decisions.
   
2. Risk-Based Borrower Segmentation Rather than treating all borrowers equally, lenders should classify applicants into different risk categories such as low, medium, and high-risk based on the predicted probability of default. This would allow institutions to:
       • Offer lower interest rates and higher loan limits to low-risk borrowers,
       • Apply stricter conditions or higher interest rates to medium-risk borrowers, and
       • Limit exposure to high-risk borrowers or require additional guarantees. This approach supports both financial                inclusion and portfolio sustainability.
   
3. Use of the Model as an Early Warning System by periodically updating borrower data, the institution can identify customers whose risk of default is increasing and take early action, such as sending reminders, offering financial guidance, or restructuring loans. This can help reduce non-performing loans.
   
4. Integration with Human Decision-Making Although the model provides valuable insights, it should be used as a decision-support tool rather than a complete replacement for loan officers. Human judgment is still important, especially for borderline cases or for assessing qualitative factors such as business viability and local market conditions.
   
5. Continuous Model Improvement The accuracy of the model can be further improved by incorporating more detailed borrower information, such as business cashflows, mobile money transaction history, and past repayment behavior. Regular retraining of the model with new data will also ensure that it remains relevant and reliable over time.

**DEPLOYMENT**
Have a look at our model deployment

https://loan-default-risk-detection-j3gtsrd8mmpkygkf8mup3r.streamlit.app/








