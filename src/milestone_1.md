# **Milestone 1: Project Plan**
## NBA Game Outcome Prediction Using Bayesian Networks

---

## **Problem Description**

Being able to predict the outcomes of NBA games is a multi-layered question that if modeled correctly, can make you millions. It involves many confounding variables and factors such as team strength and talent, player injuries, team momentum, and contextual game information. Traditional machine learning models often treat each of these factors as separate variables, independent features. However, these features are inherently causal and therefore it is important to look deeper at the underlying connections between these analytics and the night-to-night uncertainty on the basketball court.

**Research Question:**  
Is there a Bayesian Network that accurately models the relationships between team performance, context, and wins/losses?

**Objectives:**
- Create a Bayesian Network that models dependencies between game factors
- Create models with prediction accuracy > baseline
- Find out how much specific factors/features impact win probability

---

## **Data**

**Source:** Basketball Reference NBA Historical Data  
[kaggle.com/datasets/wyattowalsh/basketball/data](https://kaggle.com/datasets/wyattowalsh/basketball/data)

**Characteristics:**
- **Time range:** 2003-2024 seasons (modern NBA era with consistent data tracking and relevant basketball trends)
- **Size:** Approximately 20,000 regular season games after filtering out early season games (first 10 games of each team's season excluded to ensure sufficient rolling statistics)
- **Target variable:** Balanced outcome variable (the home team wins ~58% historically)
- **Data quality:** Minimal missing data in core stats

---

## **Methodology**

### **Data Preprocessing & Feature Engineering**
- Temporal Filtering
- Temporal split of Train, Validation, and Test Data
- Calculate team stats
- Context Features
- Advanced metrics
- Discretization

### **Bayesian Network Structure Design**

**Variables:**
- **Root nodes:** Season-long cumulative team statistics
- **Intermediate nodes:** Recent form, rest days, roster health
- **Child nodes:** Matchup advantage, home court factor
- **Target:** Game outcome

**Structure:** DAG based on basketball domain knowledge (Team strength → Recent form → Matchup advantage → Outcome, etc.)

### **Model Training and Inference**

**Python Packages:** pgmpy or pomegranate for Bayesian networks

**Hyperparameter learning:** Can introduce Bayesian estimation to tune parameters and prevent overfitting

**Inference:** Variable elimination for probabilistic predictions over deterministic ones

### **Evaluation Metrics**

**Classification:**
- Accuracy
- Precision/Recall
- F1-score
- ROC-AUC

**Probabilistic:**
- Log-loss

**Baseline:** Home team heuristic and logistic regression model serve as lightweight models to serve as baselines
