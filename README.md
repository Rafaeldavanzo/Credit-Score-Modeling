Analysis and Prediction of Customer Default Rates

Objective:
To analyze customer default rates, develop predictive models for defaults using statistical and machine learning techniques, and analyze customer feedback for insights.

Phases:

    Data Management:
        Tasks:
            Read and merged multiple datasets (customer profiles, info, and credit details).
            Cleaned data by handling duplicates, correcting categorical values, and managing missing values.
            Converted continuous variables into categorical variables.
            Calculated the overall default rate.
        Outcome: Created a comprehensive and clean master dataset, ready for detailed analysis.

    Exploratory Data Analysis (EDA):
        Tasks:
            Generated box-whisker plots for numeric variables by default status.
            Created tables showing default rates for categorical variables (e.g., gender, zone).
            Produced bar charts to compare means of numeric variables for defaulters vs. non-defaulters.
            Developed heatmaps to show bad rates by zone and credit debt category.
        Outcome: Identified key patterns and insights, such as higher default rates associated with higher debt-to-income ratios.

    Logistic Regression Model:
        Tasks:
            Partitioned data into training and test sets.
            Ran initial logistic regression, refined based on significant variables, and checked for multicollinearity.
            Evaluated model performance using ROC curves, AUC, and confusion matrices.
            Found optimal threshold to balance sensitivity and specificity.
        Outcome: Developed a logistic regression model with good predictive capabilities for customer defaults.

    Machine Learning Models:
        Tasks:
            Applied Naive Bayes, Decision Tree, and Random Forest models on training data.
            Evaluated each model using ROC curves, AUC, confusion matrices, and accuracy metrics.
            Visualized the Decision Tree for better interpretability.
        Outcome: Random Forest emerged as the best model, balancing accuracy and AUC effectively.

    Customer Feedback Analysis:
        Tasks:
            Pre-processed customer feedback text for analysis.
            Generated word clouds to visualize common terms.
            Conducted sentiment analysis and visualized sentiment score distributions.
            Identified the top 10 most mentioned words and visualized them using bar and pie charts.
        Outcome: Majority of feedback was positive. Identified key areas for improvement and customer sentiments.

Key Findings:

    Default Rates: Defaulters had higher debt-to-income ratios and credit debt, indicating higher risk associated with increased debt.
    Best Model: The Random Forest model provided the best predictive performance with balanced accuracy and AUC.
    Customer Feedback: Feedback analysis revealed predominantly positive sentiments, with frequent mentions of terms like "loan," "bank," and "service," indicating areas of focus for customer satisfaction.

This project successfully combined data management, exploratory data analysis, predictive modeling, and customer sentiment analysis to provide a comprehensive understanding of customer default behaviors and insights into improving customer satisfaction.
