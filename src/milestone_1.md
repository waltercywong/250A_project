# **Milestone 1: Project Plan**
## NBA Game Outcome Prediction Using Bayesian Networks

---

### **Problem Description**

Predicting NBA game outcomes is a complex problem involving multiple interacting factors including team strength, player availability, contextual game situations, and momentum. Traditional machine learning models treat these factors independently, but basketball outcomes are inherently causal—team performance depends on roster health, which depends on schedule density, which affects game outcomes. 

**Research Question:** Can we build an interpretable Bayesian Network that models the causal relationships between team performance metrics, contextual factors (home advantage, rest days, roster health), and game outcomes while providing probabilistic predictions and insights into key winning factors?

**Objectives:**
1. Develop a Bayesian Network that captures causal dependencies between game factors
2. Achieve prediction accuracy exceeding baseline models (home team heuristic ~58%, logistic regression ~65%)
3. Quantify the impact of specific factors (e.g., rest days, home court advantage, roster health) on win probability
4. Provide interpretable insights into what drives NBA game outcomes

---

### **Dataset Source**

**Primary Dataset:** Basketball Reference NBA Historical Data (Kaggle: `wyattowalsh/basketball`)

**Key Tables:**
- `game.csv` (~65,000 games, 1946-2023): Game results, team statistics (FG%, rebounds, assists, turnovers), final scores, win/loss records
- `game_info.csv`: Game metadata (date, attendance, game time)
- `inactive_players.csv`: Player availability per game (roster health proxy)
- `officials.csv`: Referee assignments per game
- `other_stats.csv`: Advanced game statistics (paint points, fast break points, turnovers)
- `team.csv` & `team_history.csv`: Team information and historical data
- `common_player_info.csv` & `player.csv`: Player information for roster analysis

**Data Characteristics:**
- Time range: 2003-2024 seasons (modern NBA era with consistent defensive statistics and pace-adjusted metrics)
- ~20,000 regular season games after filtering early season games (first 10 games of each team's season excluded to ensure sufficient rolling statistics)
- Balanced outcome variable (home team wins ~58% historically)
- Minimal missing data in core statistics; strategic imputation for advanced metrics

---

### **Methodology**

**1. Data Preprocessing & Feature Engineering**
- **Temporal filtering:** Only games after 09-01-2003 (2003-04 season onwards); exclude first 10 games of each team's season to ensure sufficient rolling statistics
- **Temporal split:** Train (2003-2018), Validation (2019-2021), Test (2022-2024) to avoid data leakage
- **Calculate team statistics from `game.csv`:** Compute rolling/cumulative metrics (win%, PPG, FG%, rebounds, assists) from historical game results for each team before each game
- **Contextual features:** Days of rest (calculated from `game_info.csv` dates), back-to-back indicator, home/away status, number of inactive players
- **Recent form:** Last 5-game and 10-game win percentages, scoring trends
- **Advanced metrics from `other_stats.csv`:** Points in paint, fast break points, second chance points
- **Discretization:** Bin continuous variables into categories (e.g., team strength: Weak/Average/Strong; rest: 0-days/1-day/2-days/3+)

**2. Bayesian Network Structure Design**
- **Variables:**
  - Root nodes: Season-long team statistics (offensive/defensive ratings, win%)
  - Intermediate nodes: Recent form, rest days, roster health
  - Child nodes: Matchup advantage, home court factor
  - Target: Game outcome (binary: home win/loss)
  
- **Structure:** Expert-driven DAG based on basketball domain knowledge:
  - Team strength → Recent form → Matchup advantage → Outcome
  - Rest days → Performance capacity → Outcome
  - Roster health → Team effectiveness → Outcome
  - Home court advantage → Outcome (direct effect)

**3. Model Training & Inference**
- **Tool:** pgmpy (Python) for Bayesian Network implementation
- **Parameter learning:** Bayesian estimation with Dirichlet priors (BDeu) to prevent overfitting
- **Inference:** Variable Elimination for probabilistic predictions
- **Alternative:** Compare with structure learning algorithms (Hill Climbing + BIC score) to validate expert structure

**4. Evaluation Metrics**
- **Classification:** Accuracy, Precision, Recall, F1-score, ROC-AUC
- **Probabilistic:** Brier score (calibration quality), log-loss
- **Baselines:** Home team heuristic, team strength differential, logistic regression, random forest
- **Stratified analysis:** Performance by team strength differential, season stage, rest advantage

**5. Model Interpretation**
- Examine Conditional Probability Tables (CPDs) to understand learned relationships
- Sensitivity analysis: quantify impact of each factor on win probability
- Markov blanket analysis: identify minimal sufficient feature set
- Counterfactual reasoning: "what if" scenarios (e.g., effect of additional rest day)

**Expected Outcomes:**
- Predictive accuracy: 68-72% (target)
- Interpretable causal model with quantified effects
- Insights into non-obvious factors affecting NBA game outcomes