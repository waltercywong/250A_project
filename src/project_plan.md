## **Detailed Process Outline: Game Outcome Prediction with Bayesian Networks**

---

## **Phase 1: Data Understanding & Exploration**

### **1.1 Dataset Inventory**
**Objective:** Understand available data and relationships

**Key files:**
- `game.csv` - Main game results with team stats
- `game_summary.csv` - Game metadata (date, attendance, time)
- `game_info.csv` - Additional game information
- `line_score.csv` - Quarter-by-quarter scores
- `team_info_common.csv` - Season-level team statistics
- `inactive_players.csv` - Player availability
- `officials.csv` - Referee assignments
- `other_stats.csv` - Advanced game statistics

**Tasks:**
```python
# Explore data dimensions
- Check number of games (rows)
- Identify time range (seasons covered)
- Check for missing values
- Understand home vs away structure
- Identify unique teams and seasons
```

### **1.2 Exploratory Data Analysis**
**Create visualizations:**
- Home win percentage over time
- Distribution of point differentials
- Team performance trends
- Missing data patterns
- Correlation analysis of game statistics

**Key questions:**
- What's the baseline home win percentage?
- How predictable are game outcomes?
- Which features have the most variance?
- Are there temporal trends or regime changes?

---

## **Phase 2: Feature Engineering**

### **2.1 Target Variable**
**Define outcome variable:**
```python
target = "home_team_wins"  # Binary: 1 = home wins, 0 = away wins
# Alternative: point_differential (continuous, for regression)
```

### **2.2 Team Performance Features**
**From `team_info_common.csv` - aggregate by season:**

**Current season performance (before game):**
- Win percentage (rolling)
- Points per game (offense strength)
- Opponent points per game (defense strength)
- Offensive rating
- Defensive rating
- Net rating
- Conference rank
- Division rank

**Recent form (last N games):**
- Win percentage last 5 games
- Win percentage last 10 games
- Scoring trend
- Plus/minus trend

**Advanced metrics:**
- Pace (possessions per game)
- True shooting percentage
- Effective field goal percentage
- Turnover rate
- Offensive/defensive rebound percentage

### **2.3 Head-to-Head History**
**From `game.csv`:**
- Historical win percentage (Team A vs Team B)
- Average point differential in past matchups
- Home team advantage in this matchup
- Games played against each other this season

### **2.4 Contextual Features**

**Game situation:**
- Home vs Away (binary)
- Days of rest (calculate from game dates)
- Back-to-back games (binary)
- Travel distance (calculate from city locations)
- Time of season (early/mid/late, playoff implications)
- Weekend vs weekday
- National TV game (if available)

**Roster health:**
- Number of inactive players (from `inactive_players.csv`)
- Key player out (if identifiable)

**External factors:**
- Referee crew (from `officials.csv`) - encode as categorical or use historical bias
- Attendance (from `game_info.csv`) - proxy for home court advantage
- Arena altitude (Denver effect)

### **2.5 Derived Features**

**Strength differential:**
```python
# Create pairwise comparisons
offensive_advantage = home_ppg - away_opp_ppg
defensive_advantage = home_opp_ppg - away_ppg
pace_differential = home_pace - away_pace
```

**Momentum features:**
```python
home_streak = consecutive_wins_or_losses
away_streak = consecutive_wins_or_losses
form_difference = home_last5_win% - away_last5_win%
```

---

## **Phase 3: Bayesian Network Structure Design**

### **3.1 Define Variable Types**

**Outcome (Child node):**
- `game_outcome` - Binary (Home Win/Loss)

**Parent nodes (causes):**

**Tier 1: Team Quality (season-long attributes)**
- `home_team_strength` - Discretized (Weak/Average/Strong)
- `away_team_strength` - Discretized (Weak/Average/Strong)
- `home_offensive_rating` - Continuous or discretized
- `home_defensive_rating` - Continuous or discretized
- `away_offensive_rating` - Continuous or discretized
- `away_defensive_rating` - Continuous or discretized

**Tier 2: Recent Form (short-term state)**
- `home_recent_form` - Discretized (Poor/Average/Good)
- `away_recent_form` - Discretized (Poor/Average/Good)
- `days_rest_home` - Discretized (0, 1, 2, 3+)
- `days_rest_away` - Discretized (0, 1, 2, 3+)

**Tier 3: Game Context**
- `home_court_advantage` - Binary or continuous
- `matchup_history` - Discretized (Home favored/Even/Away favored)
- `season_stage` - Categorical (Early/Mid/Late/Playoff)
- `roster_health_home` - Discretized (Full/Depleted)
- `roster_health_away` - Discretized (Full/Depleted)

### **3.2 Network Structure**

**Proposed DAG (Directed Acyclic Graph):**

```
                 Team Season Stats
                 /              \
                /                \
        Home_Strength      Away_Strength
               |                  |
               |                  |
        Home_Off_Rating    Away_Off_Rating
        Home_Def_Rating    Away_Def_Rating
               |                  |
               v                  v
        Home_Recent_Form   Away_Recent_Form
               |                  |
               |                  |
        Days_Rest_Home     Days_Rest_Away
               |                  |
               \                  /
                \                /
                 \              /
                  v            v
              Matchup_Advantage
                      |
                      v
              Home_Court_Factor
                      |
                      v
                 Game_Outcome
```

**Key conditional dependencies:**
1. `Game_Outcome` depends on:
   - `Matchup_Advantage` (direct)
   - `Home_Court_Factor` (direct)
   - `Roster_Health` (both teams)

2. `Matchup_Advantage` depends on:
   - `Home_Recent_Form`
   - `Away_Recent_Form`
   - `Days_Rest` (both teams)
   - Historical matchup

3. `Recent_Form` depends on:
   - `Team_Strength` (season stats)
   - Schedule difficulty

### **3.3 Structure Learning Approaches**

**Option A: Expert-driven (recommended initially)**
- Define structure based on basketball domain knowledge
- Ensure causal relationships make sense
- Validate with correlation analysis

**Option B: Data-driven**
- Use Hill Climbing algorithm
- Use PC algorithm
- Use constraint-based methods
- Compare learned structure to expert knowledge

**Option C: Hybrid**
- Start with expert structure
- Allow data to refine edge weights
- Use structure learning with constraints (blacklist impossible edges)

---

## **Phase 4: Data Preprocessing**

### **4.1 Temporal Considerations**

**Critical: Avoid data leakage!**
```python
# For each game, use only information available BEFORE the game
# - Team stats: cumulative through previous games
# - Recent form: based on last N completed games
# - Head-to-head: only past matchups
```

**Create train/validation/test split by time:**
```python
# Option 1: By season
train: seasons 1996-2015
validation: seasons 2016-2018
test: seasons 2019-2023

# Option 2: Rolling window
for each season:
    train: all previous seasons
    test: current season
```

### **4.2 Feature Discretization**

**For categorical Bayesian networks, discretize continuous variables:**

```python
# Team strength (based on win percentage)
bins = [0, 0.35, 0.50, 0.65, 1.0]
labels = ['Weak', 'Below_Avg', 'Above_Avg', 'Strong']

# Offensive/defensive rating
bins = [percentile_25, percentile_50, percentile_75]
labels = ['Poor', 'Average', 'Good', 'Elite']

# Days rest
bins = [0, 1, 2, 3, inf]
labels = ['B2B', '1_day', '2_days', '3+_days']
```

**Alternatively: Use Gaussian Bayesian Networks for continuous variables**

### **4.3 Handling Missing Data**

**Strategies:**
```python
# Missing team stats early in season
- Use previous season stats
- Use league average
- Impute with team history

# Missing inactive player data
- Assume full roster if missing
- Use indicator variable for "data_available"

# Missing officials/attendance
- Create "unknown" category
- Impute with season average
```

### **4.4 Feature Scaling**

**For continuous variables in Gaussian BN:**
```python
# Standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
continuous_features = scaler.fit_transform(train_data)
```

---

## **Phase 5: Model Building**

### **5.1 Tool Selection**

**Python libraries:**
```python
# Option 1: pgmpy (most comprehensive for BN)
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.inference import VariableElimination

# Option 2: pomegranate (faster, good API)
from pomegranate import BayesianNetwork

# Option 3: bnlearn (structure learning focused)
import bnlearn as bn
```

### **5.2 Define Network Structure**

**Using pgmpy:**
```python
from pgmpy.models import BayesianNetwork

# Define structure
model = BayesianNetwork([
    # Team strength nodes
    ('home_strength', 'home_recent_form'),
    ('away_strength', 'away_recent_form'),
    
    # Recent form affects matchup
    ('home_recent_form', 'matchup_advantage'),
    ('away_recent_form', 'matchup_advantage'),
    
    # Rest affects performance
    ('days_rest_home', 'home_recent_form'),
    ('days_rest_away', 'away_recent_form'),
    
    # Context factors
    ('matchup_advantage', 'game_outcome'),
    ('home_court_factor', 'game_outcome'),
    ('roster_health_home', 'game_outcome'),
    ('roster_health_away', 'game_outcome'),
    
    # Direct effects
    ('home_strength', 'game_outcome'),
    ('away_strength', 'game_outcome')
])
```

### **5.3 Structure Learning (Alternative)**

**Learn structure from data:**
```python
from pgmpy.estimators import HillClimbSearch, BicScore

# Learn structure
hc = HillClimbSearch(train_data)
best_model = hc.estimate(scoring_method=BicScore(train_data))

print("Learned edges:", best_model.edges())
```

**With constraints:**
```python
# Blacklist impossible edges (e.g., outcome can't cause team strength)
blacklist = [('game_outcome', 'home_strength'), 
             ('game_outcome', 'away_strength')]

# Whitelist must-have edges
whitelist = [('home_strength', 'game_outcome')]

best_model = hc.estimate(
    scoring_method=BicScore(train_data),
    black_list=blacklist,
    white_list=whitelist
)
```

### **5.4 Parameter Learning**

**Fit the model to training data:**

```python
from pgmpy.estimators import MaximumLikelihoodEstimator

# Maximum Likelihood Estimation
model.fit(train_data, estimator=MaximumLikelihoodEstimator)

# OR Bayesian Parameter Estimation (with priors)
from pgmpy.estimators import BayesianEstimator

model.fit(
    train_data, 
    estimator=BayesianEstimator,
    prior_type='BDeu',  # Bayesian Dirichlet equivalent uniform
    equivalent_sample_size=10
)
```

**View learned CPDs (Conditional Probability Distributions):**
```python
for cpd in model.get_cpds():
    print("CPD of", cpd.variable)
    print(cpd)
```

---

## **Phase 6: Inference & Prediction**

### **6.1 Setup Inference Engine**

```python
from pgmpy.inference import VariableElimination

inference = VariableElimination(model)
```

### **6.2 Make Predictions**

**For a single game:**
```python
# Evidence: observed variables for a specific game
evidence = {
    'home_strength': 'Strong',
    'away_strength': 'Average',
    'home_recent_form': 'Good',
    'away_recent_form': 'Poor',
    'days_rest_home': '2_days',
    'days_rest_away': 'B2B',
    'home_court_factor': 1,
    'roster_health_home': 'Full',
    'roster_health_away': 'Depleted'
}

# Query: probability of home team winning
result = inference.query(
    variables=['game_outcome'],
    evidence=evidence
)

print(result)
# Output: P(game_outcome=1) = 0.75, P(game_outcome=0) = 0.25
```

**Batch predictions:**
```python
predictions = []
probabilities = []

for idx, row in test_data.iterrows():
    evidence = row.drop('game_outcome').to_dict()
    
    result = inference.query(
        variables=['game_outcome'],
        evidence=evidence
    )
    
    prob_home_win = result.values[1]  # Probability of home win
    pred = 1 if prob_home_win > 0.5 else 0
    
    predictions.append(pred)
    probabilities.append(prob_home_win)
```

### **6.3 Handle Missing Evidence**

**When some features are unknown:**
```python
# Marginalize over missing variables
evidence = {
    'home_strength': 'Strong',
    'away_strength': 'Average',
    # days_rest_away is unknown - don't include in evidence
}

result = inference.query(
    variables=['game_outcome'],
    evidence=evidence
)
# Result automatically marginalizes over unknown variables
```

---

## **Phase 7: Model Evaluation**

### **7.1 Classification Metrics**

```python
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

# Basic metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

# Confusion matrix
cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(cm)

# ROC-AUC (using probabilities)
auc = roc_auc_score(y_test, probabilities)
```

### **7.2 Calibration Analysis**

**Check if predicted probabilities match actual frequencies:**
```python
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

fraction_of_positives, mean_predicted_value = calibration_curve(
    y_test, probabilities, n_bins=10
)

plt.plot(mean_predicted_value, fraction_of_positives, marker='o')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Calibration Plot')
plt.legend()
plt.show()
```

### **7.3 Brier Score**

**Measure probability accuracy:**
```python
from sklearn.metrics import brier_score_loss

brier = brier_score_loss(y_test, probabilities)
print(f"Brier Score: {brier:.4f}")
# Lower is better; 0 = perfect, 0.25 = random
```

### **7.4 Baseline Comparisons**

```python
# Baseline 1: Always predict home team wins
baseline_home = np.mean(y_test)

# Baseline 2: Predict based on team strength only
baseline_strength = (test_data['home_strength'] > test_data['away_strength']).astype(int)
baseline_acc = accuracy_score(y_test, baseline_strength)

# Baseline 3: Logistic regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_acc = lr.score(X_test, y_test)

print(f"Home team baseline: {baseline_home:.3f}")
print(f"Strength baseline: {baseline_acc:.3f}")
print(f"Logistic Regression: {lr_acc:.3f}")
print(f"Bayesian Network: {accuracy:.3f}")
```

### **7.5 Stratified Analysis**

**Performance by subgroups:**
```python
# By strength differential
strong_home_games = test_data[test_data['home_strength'] == 'Strong']
weak_away_games = test_data[test_data['away_strength'] == 'Weak']

# By season stage
early_season = test_data[test_data['season_stage'] == 'Early']
late_season = test_data[test_data['season_stage'] == 'Late']

# By rest advantage
rest_advantage = test_data[test_data['days_rest_home'] > test_data['days_rest_away']]
```

---

## **Phase 8: Model Interpretation**

### **8.1 Conditional Probability Tables**

**Examine learned relationships:**
```python
# Most important node: game_outcome
cpd_outcome = model.get_cpds('game_outcome')
print(cpd_outcome)

# Analyze: How does home court affect win probability?
# Analyze: What's the impact of rest days?
```

### **8.2 Sensitivity Analysis**

**How much does each variable affect the outcome?**
```python
# Baseline evidence
base_evidence = {
    'home_strength': 'Average',
    'away_strength': 'Average',
    'home_recent_form': 'Average',
    'away_recent_form': 'Average',
    'days_rest_home': '2_days',
    'days_rest_away': '2_days',
    'home_court_factor': 1,
}

base_prob = inference.query(
    variables=['game_outcome'],
    evidence=base_evidence
).values[1]

# Vary each feature
for var in ['home_strength', 'days_rest_home', 'roster_health_home']:
    for value in possible_values[var]:
        modified_evidence = base_evidence.copy()
        modified_evidence[var] = value
        
        prob = inference.query(
            variables=['game_outcome'],
            evidence=modified_evidence
        ).values[1]
        
        print(f"{var}={value}: ΔProb = {prob - base_prob:.3f}")
```

### **8.3 Feature Importance**

**Mutual information with outcome:**
```python
from sklearn.metrics import mutual_info_score

for feature in features:
    mi = mutual_info_score(test_data['game_outcome'], test_data[feature])
    print(f"{feature}: MI = {mi:.4f}")
```

### **8.4 Counterfactual Analysis**

**"What if" scenarios:**
```python
# What if the away team had more rest?
evidence_actual = {...}
evidence_counterfactual = {..., 'days_rest_away': '3+_days'}

prob_actual = inference.query(['game_outcome'], evidence_actual)
prob_counter = inference.query(['game_outcome'], evidence_counterfactual)

print(f"Actual: {prob_actual}")
print(f"Counterfactual: {prob_counter}")
print(f"Effect of rest: {prob_counter - prob_actual}")
```

### **8.5 Markov Blanket Analysis**

**Find minimal set of features needed for prediction:**
```python
from pgmpy.models import MarkovNetwork

mb = model.get_markov_blanket('game_outcome')
print("Markov Blanket of game_outcome:", mb)
# These are the only variables needed for optimal prediction
```

---

## **Phase 9: Model Refinement**

### **9.1 Iterative Improvements**

**Based on evaluation results:**

1. **Add missing features:**
   - If model struggles with close games → add momentum features
   - If poor on back-to-backs → refine rest categories
   - If misses upsets → add "upset potential" variable

2. **Refine discretization:**
   - Try different binning strategies
   - Use domain-informed cutoffs
   - Experiment with number of categories

3. **Adjust structure:**
   - Add/remove edges based on learned dependencies
   - Test alternative causal pathways
   - Simplify overly complex structures

### **9.2 Ensemble Approaches**

**Combine multiple models:**
```python
# Model 1: Basic BN with season stats
# Model 2: BN with recent form emphasis
# Model 3: BN with contextual factors

ensemble_prob = (prob1 * 0.4 + prob2 * 0.3 + prob3 * 0.3)
```

### **9.3 Cross-Validation**

**Robust performance estimation:**
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
scores = []

for train_idx, test_idx in tscv.split(data):
    train_fold = data.iloc[train_idx]
    test_fold = data.iloc[test_idx]
    
    # Train model
    model.fit(train_fold)
    
    # Evaluate
    # ... predict and score
    scores.append(accuracy)

print(f"CV Accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
```

---

## **Phase 10: Deployment & Insights**

### **10.1 Create Prediction Pipeline**

```python
class GamePredictor:
    def __init__(self, model, inference_engine):
        self.model = model
        self.inference = inference_engine
    
    def predict_game(self, home_team, away_team, game_date):
        # Fetch current team stats
        home_stats = get_team_stats(home_team, game_date)
        away_stats = get_team_stats(away_team, game_date)
        
        # Create evidence
        evidence = self._prepare_evidence(home_stats, away_stats)
        
        # Predict
        result = self.inference.query(['game_outcome'], evidence)
        
        return result.values[1]  # P(home win)
```

### **10.2 Generate Insights**

**Key findings to extract:**
1. **Most important factors** for winning
2. **Home court advantage magnitude** (quantified)
3. **Rest impact** (1 day vs 2 days vs 3+)
4. **Matchup-specific patterns** (which stats matter most)
5. **Upset predictors** (when favorites lose)

### **10.3 Visualization**

**Create interpretable visuals:**
- Network structure diagram
- Feature importance bar chart
- Calibration curves
- Win probability over time (for a season)
- Conditional probability heatmaps

### **10.4 Report Structure**

```markdown
1. Introduction & Motivation
2. Data Description
3. Bayesian Network Design
   - Structure justification
   - Variable definitions
4. Methodology
   - Preprocessing
   - Parameter learning
5. Results
   - Performance metrics
   - Comparison to baselines
6. Interpretation
   - Key findings
   - Sensitivity analysis
   - Basketball insights
7. Limitations & Future Work
8. Conclusion
```

---

## **Key Deliverables Checklist**

- [ ] Clean, preprocessed dataset
- [ ] Network structure (DAG visualization)
- [ ] Trained Bayesian Network model
- [ ] Evaluation metrics (accuracy, AUC, Brier score)
- [ ] Comparison to baseline models
- [ ] Interpretation analysis (CPDs, sensitivity)
- [ ] Visualization of key findings
- [ ] Final report with basketball insights
- [ ] Code repository with documentation

---

## **Expected Challenges & Solutions**

| Challenge | Solution |
|-----------|----------|
| **Data leakage** | Strict temporal train/test split |
| **Class imbalance** | Home teams win ~60% → use balanced metrics |
| **Missing data** | Careful imputation + indicator variables |
| **Overfitting** | Cross-validation + Bayesian priors |
| **Computational cost** | Use discrete variables, optimize structure |
| **Interpretability** | Focus on small, meaningful networks |

---

## **Timeline Estimate**

- **Week 1:** Data exploration, cleaning, feature engineering
- **Week 2:** Initial BN structure design, discretization, train/test split
- **Week 3:** Model training, parameter learning, baseline comparisons
- **Week 4:** Evaluation, interpretation, sensitivity analysis
- **Week 5:** Refinement, visualization, report writing

