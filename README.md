# CKD

🧠 Predicting Chronic Kidney Disease Progression Using a Hybrid Ensemble ML Model
Chronic Kidney Disease (CKD) is a progressive condition where early detection and accurate prediction can drastically improve patient outcomes. In this project, I developed an ensemble-based ML system to predict CKD progression by integrating multiple algorithms and incorporating clinical insights into the model architecture. The goal was not just accuracy, but interpretability and reliability in high-stakes medical decision-making.

🛠 Project Motivation
CKD often goes undiagnosed in its early stages due to vague symptoms and inconsistent testing patterns. Existing machine learning models often lack robustness when handling real-world, incomplete, or noisy data — a common scenario in clinical settings. I wanted to create a model that adapts to such challenges while maintaining high accuracy.

🔍 Dataset & Preprocessing
The dataset included over 400 patient records with features such as:

Hemoglobin level

Serum Creatinine

Specific Gravity (SG)

Albumin

Blood Pressure, etc.

Preprocessing involved:

Missing value imputation using domain-aware strategies (e.g., median for SG, zero-imputation for Albumin).

Normalization for numeric features to improve model convergence.

One-hot encoding for categorical variables like hypertension and diabetes.

🧩 Model Architecture: Adaptive Cluster-Weighted Decision Forest (ACWDF)
The core of the system was a hybrid ensemble combining:

XGBoost for capturing non-linear patterns,

Random Forest for robustness against noise,

SVM for edge-case boundary handling.

A custom-designed dynamic weighting system adjusted model contributions based on the reliability of clinical indicators. For example:

In cases with low hemoglobin and high creatinine, the XGBoost model's weight increased.

For borderline cases, SVM had a larger say in the final decision.

🤖 Uncertainty Quantification
To enhance trust in predictions, I included an uncertainty module using a shallow neural network. This network flagged conflicting or low-confidence predictions, giving clinicians insight into when human oversight is needed.

📈 Results
Accuracy: ~95.2%

F1 Score: 0.93

ROC-AUC: 0.96

Flagged 7% of predictions as uncertain, which correlated strongly with actual edge-case records.

🌟 Key Takeaways
Hybrid ensemble methods are effective for handling noisy, imbalanced medical data.

Custom weighting logic improves prediction interpretability and trust.

Adding an uncertainty layer allows for better clinical decision support, a step toward real-world deployability.

🧩 Future Work
Deploy the model as a web-based clinical tool with a Flask backend.

Explore time-series CKD progression using LSTM or Transformer models.

Collaborate with a nephrologist for real-world validation and feedback.
