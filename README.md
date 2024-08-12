# web-traffic-classification

## Overview
This project focuses on building a robust machine learning model to classify web traffic into categories such as 'Browser', 'Robot', 'Hacker', and 'Other'. The classification model is designed with an **MLOps Mindset**, ensuring that the end-to-end process, from feature engineering to model serving, is optimized for real-world deployment. The primary goal is to develop a model that can be used for real-time inference, enabling proactive interventions in suspicious web traffic sessions.

## One Key Mindset 
**"MLOps Mindset"** -  Keep in mind how the model will be served in the end from the beginning. Use that mindset as guide to construct features.

The model can be served for (offline) batch analysis or real-time inference. I am more intereted in building a ML application that can be deployed for real senerio, therefore I choose real-time inference, and all the session related features are cummulative. 

#### Real-Time Inference Use Case
 For real-time inference, the model detects suspicious traffic during ongoing sessions. When traffic is flagged as suspicious (e.g., identified as a robot), an intervention mechanism is triggered before the session ends. This could include a challenge such as a CAPTCHA ("I am not a robot" checkbox or selecting images). Failure to pass the challenge results in the session being blocked, protecting the website from potential threats.

#### Feature Construction for Real-Time Detection
Given the focus on real-time detection, session-related features are designed to be cumulative, reflecting ongoing user behavior. This approach contrasts with offline analysis, where session-level features might encompass the entire session retrospectively.

## Three Tricks for Handling Imbalanced Multi-Class Classification 

1. **Resampling techniques**: 
    - Oversampling, downsampling. E.g. data replication; SMOTE(synthetic minority oversampling); add synthetic data for images by rotating, chopping, etc. 
    - However in out case, resampling was avoided as the minority classes are not the primary focus.
2. **Model performance metrics**: 
    - should not be accuracy, but F1 score or AU-PRC (instead of AU-ROC)
3. **Stratified splitting**: 
    - Stratified splitting ensures that the distribution of classes remains consistent across training, validation, and test sets, which is critical for reliable model evaluation. This strategy is also applied during cross-validation.

For more detailed insights, check out my [blog post](https://medium.com/@SiqiLi/imbalanced-dataset-in-classification-a28e564124d5).

The original dataset contained 8 distinct classes in the `ua_agent_class` column. To simplify the problem and improve model performance, the classes were consolidated as follows:
- Browser: Combined 'Browser' and 'Browser Webview'.
- Robot: Combined 'Robot' and 'Robot Mobile'.
- Other: Combined 'Special', 'Mobile App', and 'Cloud Application'.

Resulting class distribution:
```
Browser    37309
Robot      21141
Hacker      1177
Other        155
```

## Two Conditions for Data Splitting
1. **Session-Based Splitting**: Ensures that data from the same session is not split between training and test sets, preserving the temporal context and avoiding data leakage.
2. **Stratified Splitting**: Maintains the class distribution across different splits, critical for dealing with the imbalanced nature of the dataset.

These conditions are consistently applied, including during cross-validation for feature importance analysis and hyperparameter tuning.

## Features
### Feature Constuction

#### Time Features

- `epoch_ms` can be transformed to datetime format. Pertentially useful to extract time related features, such as year, month, day, hour, day_of_week, week_of_year, etc. 
    - However, this dataset spans only two hours on a single day. As a result, these features were not useful.
    - **Recommendation**: Future data collection should span various time ranges to build a more generalized model.
    - **Caution**: Sessions might be incomplete due to the limited time span of the data.

#### Session Features
Several session-related features were engineered to capture user behavior dynamically:
- `time_diff`: Time differences between consecutive requests in the same session
- `cumulative_session_duration`
- `cumulative_num_unique_visitor_recognition_type`
- `cumulative_num_requests` 
- `cumulative_num_unique_pages_visited`
- `cumulative_avg_time_between_requests`
- `cumulative_count_referer`
- `cumulative_num_unique_referer`

After performing correlation analysis and feature importance evaluation, `cumulative_num_unique_pages_visited` and `cumulative_num_unique_visitor_recognition_type` were excluded from the final model.

#### URL Features

Top-k frequent terms from `url_without_parameters` were identified and used to create one-hot encoded features (`contains_*`) indicating the presence of these terms in the URL.

- **Tuning `k`**: After tuning in the [1_feature notebook](notebooks/1_feature.ipynb), k=25 was found to be the optimal trade-off between model performance and complexity.
- **Dynamic Adaptation**: These features are designed to adapt to data drift, ensuring the model remains effective over time.

Two more numerical features are constructed:
- `url_length`
- `url_depth`

#### Other Categorical Features
- `country_by_ip_address`: high cardinality, frequency encoded to `country_frequency_encoded`
- `region_by_ip_address`: high cardinality, frequency encoded to `region_frequency_encoded`
- `visitor_recognition_type`: low cardinlity, one-hot encoded

### Feature Selection
- For numerical features, apply *correlation analysis* between them, then keep only one feature among those highly correlated features.

- For categorical features, apply *chi-square analysis* between this feature and the target to assess associations. However, high association != high pridictive power, thus apply

- *Feature important anaysis*

## Model Selection: RandomForestClassifier
Why RandomForest?
- Ensemble model
    - better performance than a single learner
    - more resilient towards overfitting
- Tree-based
    - easy feature importance analysis
    - easy feature engineering
        - tree-based + frequency encoding for high cardinality features 
        - no need to scale or standadized numerical features

For more on frequency encoding, check out my [blog post](https://medium.com/@SiqiLi/frequency-encoding-4156b92e7942).


## Model Serving 
To ensure consistency during model serving, it's crucial to replicate the feature engineering process used during training:
- **Encoding**: Save the frequency encoding mappings and one-hot encoder during training to apply to new data during inference.
- **Top-k Terms**: Save the identified top-k terms for URL feature extraction.
- **Consistency**: Ensure that all feature transformations applied during training are identically applied during serving.

## Feedback Loop and Continuous Improvement

The model's predictions will be continuously evaluated against real-world data, such as feedback from CAPTCHA challenges ("I am not a robot"). This feedback will be used to calculate precision, especially for "Robot" predictions.

How precision is calculated w.r.t. Robot:

For an ANONYMOUS traffic, once it is predicted to be Robot, if the 'user' can indicate to be a human, then it would be FP (false positive); if the user cannot pass the CAPTCHA challenge, then it would be TP (true positive). Calculate the precision = TP/(TP+FP). We want a higher precision.

### Continuous Monitoring
- **Precision Monitoring**: Track precision, particularly for the "Robot" class. If precision drops below a threshold, the model will be retrained.
- **F1 Score Monitoring**: Similarly, monitor the F1 score to ensure overall model performance remains high.
- **Dynamic Feature Updates**: As the dataset becomes more diverse (e.g., by collecting data across different times), time-related features will be revisited. URL-based features are dynamically adapted to account for data drift.

