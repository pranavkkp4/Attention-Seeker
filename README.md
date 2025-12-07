```markdown
Attention Seeker  
A Machine Learning Framework for Quantifying Human Attention Using Wearable Sensor Data  
Authors: Pranav Kaliaperumal & Kevin Jacob



Overview
Attention Seeker is a full machinelearning pipeline designed to estimate a user’s attention level using physiological and motion data collected from wearable sensors. The project computes:

 A computed Attention Score using heart rate, HRV approximation, and wrist movement  
 A modeled Outside Factors Score (sleep, screen time)  
 Pearson correlation between attention and outside factors  
 Regression and classification models predicting cognitive load and attention lapses  
 Train/test datasets for reproducible ML experimentation  

The project is implemented entirely in Python using Jupyter Notebooks.



 Repository Structure
```

AttentionSeeker/
│
├── data/
│   └── merged_sensors.csv               Wearable dataset (source data)
│
├── notebooks/
│   ├── attention_seeker_analysis.ipynb          Exploratory prototype notebook
│   ├── attention_seeker_data_analysis.ipynb     Final analysis & ML results
│   └── attention_seeker_train_test_pipeline.ipynb  Reproducible ML pipeline
│
├── processed/
│   ├── attention_scores.csv             Final dataset with computed features
│   ├── train.csv                        Training dataset
│   └── test.csv                         Testing dataset
│
├── attention_seeker_final_project.zip   Full packaged project (optional)
└── README.md

```



 Purpose of Each Notebook

 1. `attention_seeker_analysis.ipynb` — Exploratory Notebook
Used for:
 Inspecting and understanding raw dataset  
 Testing feature engineering functions  
 Debugging timestamp alignment and HRV approximation  
 Early prototype of the Attention Score  

Not part of the final reproducible pipeline.



 2. `attention_seeker_data_analysis.ipynb` — Final Analysis Notebook
Contains:
 Data preprocessing and feature extraction  
 Attention Score computation  
 Outside Factors modeling  
 Correlation analysis  
 Regression and classification ML models  
 All visualizations and evaluation metrics  

This is the main research notebook for analysis and results.



 3. `attention_seeker_train_test_pipeline.ipynb` — Reproducible Pipeline
Contains:
 Deterministic dataset preprocessing  
 Feature generation (HR, HRV, movement, Attention Score, Outside Factors)  
 Split into train/test sets  
 Exports `train.csv`, `test.csv`, and `attention_scores.csv`  

Designed for reproducible experiments separate from the analysis notebook.



 Attention Score Formula

```

HR  = (HR_t   HR_rest)  / HR_rest
HRV = (HRV_t  HRV_rest) / HRV_rest
M   = (M_rest  Movement_t) / M_rest

```

Weighted sum:

```

AttentionScore = 0.25  HR + 0.50  HRV + 0.25  M

```

Scores > 0 → attentive  
Scores < 0 → inattentive / possible lapse  



 Outside Factors Score

```

SleepHours  ~ N(7.5, 0.7)
ScreenTime  ~ N(3.5, 1.0)
OF = (Sleep  7.5)  (ScreenTime  3.5)

```

Positive = good routines  
Negative = harmful routines  



 Correlation Analysis

Pearson correlation:

```

r = SUM((AS  mean(AS))  (OF  mean(OF))) /
sqrt( SUM((AS  mean(AS))^2)  SUM((OF  mean(OF))^2) )

```



 Installation

Install dependencies:

```

pip install pandas numpy scikitlearn matplotlib seaborn

```



 How to Run the Project

 1. Generate Train/Test Datasets
Open:

```

notebooks/attention_seeker_train_test_pipeline.ipynb

```

Run all cells.  
Outputs: `attention_scores.csv`, `train.csv`, `test.csv`



 2. Run Full ML Analysis
Open:

```

notebooks/attention_seeker_data_analysis.ipynb

```

This performs:
 Feature engineering  
 Correlation analysis  
 Regression & classification modeling  
 Visualization generation  



 3. (Optional) Review Prototype Notebook
```

notebooks/attention_seeker_analysis.ipynb

```

Explores dataset behavior and early prototypes.



 Machine Learning Components

 Regression  
Predict continuous Attention Score  
Metrics:
 R²  
 MAE  

 Classification  
Predict attention lapses  
Models:
 Logistic Regression  
 Random Forest  

Metrics:
 Accuracy  
 F1  
 ROC AUC  



 Project Authors
Pranav Kaliaperumal  
Kevin Jacobs  

University of Colorado Denver — Machine Learning  



 License
For academic and research purposes only.  
Dataset subject to original authors’ license.
```
