I'll provide a detailed breakdown of what your final preprocessed dataframe will look like after all the data engineering steps.

## **Final Dataframe Structure**

### **High-Level Overview**
- **Rows:** ~16,000-17,000 games (2003-2024 regular season, post 09-01-2003, after excluding first 10 games of each team's season)
- **Columns:** ~45-50 features (after feature engineering and discretization)

---

## **Column Structure by Category**

### **1. Game Identifiers (4 columns)**
```python
game_id                    int64      # Unique game identifier (e.g., 0022200001)
game_date                  datetime64 # Date of game (e.g., 2022-10-18)
home_team_id               int64      # Team ID for home team
away_team_id               int64      # Team ID for away team
```

---

### **2. Target Variable (1 column)**
```python
game_outcome               int64      # Binary: 1 = home win, 0 = away win
```

---

### **3. Home Team Features - Rolling Statistics (10 columns)**
*Computed from historical game results before this game*

```python
# Continuous (raw metrics)
home_win_pct               float64    # Cumulative win% this season (0.0-1.0)
home_ppg                   float64    # Points per game (avg: ~105-110)
home_opp_ppg               float64    # Opponent points per game (defensive strength)
home_fg_pct                float64    # Field goal percentage (0.40-0.50)
home_fg3_pct               float64    # 3-point percentage (0.30-0.40)
home_ft_pct                float64    # Free throw percentage (0.70-0.80)
home_rpg                   float64    # Rebounds per game (40-50)
home_apg                   float64    # Assists per game (20-30)
home_topg                  float64    # Turnovers per game (12-16)
home_stl_blk_pg            float64    # Steals + blocks per game (10-15)

# Discretized (for Bayesian Network)
home_team_strength         category   # 'Weak', 'Below_Avg', 'Above_Avg', 'Strong'
home_offensive_strength    category   # 'Poor', 'Average', 'Good', 'Elite'
home_defensive_strength    category   # 'Poor', 'Average', 'Good', 'Elite'
```

---

### **4. Away Team Features - Rolling Statistics (10 columns)**
*Mirror of home team features*

```python
away_win_pct               float64    
away_ppg                   float64    
away_opp_ppg               float64    
away_fg_pct                float64    
away_fg3_pct               float64    
away_ft_pct                float64    
away_rpg                   float64    
away_apg                   float64    
away_topg                  float64    
away_stl_blk_pg            float64    

# Discretized
away_team_strength         category   
away_offensive_strength    category   
away_defensive_strength    category   
```

---

### **5. Recent Form Features (6 columns)**
*Last 5 and 10 games performance*

```python
home_last5_win_pct         float64    # Win% in last 5 games (0.0-1.0)
home_last10_win_pct        float64    # Win% in last 10 games
home_scoring_trend         float64    # PPG change last 5 vs previous 5 games

away_last5_win_pct         float64    
away_last10_win_pct        float64    
away_scoring_trend         float64    

# Discretized
home_recent_form           category   # 'Poor', 'Average', 'Good'
away_recent_form           category   # 'Poor', 'Average', 'Good'
```

---

### **6. Rest & Schedule Context (6 columns)**
*Calculated from game dates*

```python
days_rest_home             int64      # Days since last game (0, 1, 2, 3+)
days_rest_away             int64      
back_to_back_home          bool       # True if 0 days rest
back_to_back_away          bool       

# Discretized
days_rest_home_cat         category   # 'B2B', '1_day', '2_days', '3+_days'
days_rest_away_cat         category   
```

---

### **7. Roster Health (4 columns)**
*From inactive_players.csv*

```python
num_inactive_home          int64      # Number of inactive players (0-5+)
num_inactive_away          int64      

# Discretized
roster_health_home         category   # 'Full' (0-1 out), 'Depleted' (2+ out)
roster_health_away         category   
```

---

### **8. Advanced Game Context (4 columns)**
*From other_stats.csv (historical averages)*

```python
home_paint_pts_pg          float64    # Points in paint per game
home_fastbreak_pts_pg      float64    # Fast break points per game
away_paint_pts_pg          float64    
away_fastbreak_pts_pg      float64    
```

---

### **9. Game Context (3 columns)**
```python
home_court_advantage       int64      # 1 for home team, 0 for neutral (always 1)
season_stage               category   # 'Early' (games 11-30), 'Mid' (31-60), 'Late' (61+)
                                      # Note: First 10 games excluded from dataset
is_weekend                 bool       # Saturday or Sunday game
```

---

### **10. Derived/Matchup Features (5 columns)**
*Computed differentials*

```python
strength_differential      float64    # home_win_pct - away_win_pct
offensive_advantage        float64    # home_ppg - away_opp_ppg
defensive_advantage        float64    # away_opp_ppg - home_opp_ppg
rest_advantage             int64      # days_rest_home - days_rest_away
form_difference            float64    # home_last5_win_pct - away_last5_win_pct

# Discretized
matchup_advantage          category   # 'Home_Favored', 'Even', 'Away_Favored'
```

---

## **Complete DataFrame Schema**

### **Continuous Features (for Gaussian BN or input)**
```python
Final DataFrame Shape: (19500, 48)

Continuous columns (30):
├── Identifiers (3): game_id, home_team_id, away_team_id
├── Home rolling stats (10): home_win_pct, home_ppg, home_opp_ppg, ...
├── Away rolling stats (10): away_win_pct, away_ppg, away_opp_ppg, ...
├── Recent form (6): home_last5_win_pct, away_last5_win_pct, ...
├── Advanced stats (4): home_paint_pts_pg, away_paint_pts_pg, ...
└── Derived features (5): strength_differential, offensive_advantage, ...

Datetime columns (1):
└── game_date

Boolean columns (4):
├── back_to_back_home
├── back_to_back_away
└── is_weekend

Integer columns (7):
├── game_outcome (TARGET)
├── days_rest_home
├── days_rest_away
├── num_inactive_home
├── num_inactive_away
├── rest_advantage
└── home_court_advantage
```

### **Discretized Features (for Discrete BN)**
```python
Categorical columns (14):
├── home_team_strength          # 4 categories
├── home_offensive_strength     # 4 categories
├── home_defensive_strength     # 4 categories
├── away_team_strength          # 4 categories
├── away_offensive_strength     # 4 categories
├── away_defensive_strength     # 4 categories
├── home_recent_form            # 3 categories
├── away_recent_form            # 3 categories
├── days_rest_home_cat          # 4 categories
├── days_rest_away_cat          # 4 categories
├── roster_health_home          # 2 categories
├── roster_health_away          # 2 categories
├── season_stage                # 3 categories
└── matchup_advantage           # 3 categories
```

---

## **Sample Rows**

```python
# Example after preprocessing
   game_id   game_date  home_team_id  away_team_id  game_outcome  home_win_pct  home_ppg  away_win_pct  away_ppg  days_rest_home  ...  home_team_strength  away_team_strength  matchup_advantage
0  22200135  2022-11-05  1610612744    1610612747             1         0.667    113.2         0.714    109.8               2  ...       'Strong'            'Strong'         'Even'
1  22200136  2022-11-05  1610612739    1610612754             0         0.429    104.5         0.571    107.3               1  ...       'Below_Avg'         'Above_Avg'      'Away_Favored'
2  22200137  2022-11-06  1610612738    1610612748             1         0.750    118.7         0.500    106.2               0  ...       'Strong'            'Average'        'Home_Favored'
```

---

## **Data Processing Pipeline Summary**

```python
# Step-by-step transformation
1. Load game.csv                           → (65000, 50) raw games
2. Filter to post 09-01-2003 (2003-2024)   → (40000, 50) modern era only
3. Filter: regular season games only       → (35000, 50) exclude playoffs
4. Join game_info.csv                      → (35000, 52) added date, attendance
5. Compute rolling statistics              → (35000, 72) added 20 rolling features
6. Calculate rest days                     → (35000, 76) added rest features
7. Join inactive_players.csv               → (35000, 80) added roster health
8. Join other_stats.csv                    → (35000, 84) added advanced stats
9. Filter: exclude first 10 games/team     → (20000, 84) removed early season
10. Create derived features                → (20000, 89) added differentials
11. Discretize for Bayesian Network        → (20000, 103) added categorical features
12. Select final feature set               → (20000, 48) keep relevant columns
```

---

## **Final Dataset Characteristics**

```python
Dataset Summary:
├── Total rows: ~19,500 games (post 09-01-2003, regular season only)
├── Training set (2003-2018): ~13,500 games (69%)
├── Validation set (2019-2021): ~3,500 games (18%)
└── Test set (2022-2024): ~2,500 games (13%)

Memory Usage:
├── Continuous features: ~5 MB (30 float64 columns)
├── Categorical features: ~1.5 MB (14 category columns)
├── Integer/bool features: ~0.8 MB (11 columns)
└── Total: ~7-8 MB (compressed)

Missing Data:
├── Early season stats: Eliminated by excluding first 10 games of each team's season
├── Inactive players: <5% missing → default to 0 (full roster)
├── Advanced stats: <2% missing → forward fill or season average
└── After preprocessing: 0% missing (all imputed)
```

---

## **Key Design Decisions**

1. **Keep both continuous and discretized versions** → Allows flexibility for Gaussian vs. Discrete BN
2. **Rolling statistics computed with expanding window** → Uses all available history, not just fixed window
3. **Discretization boundaries based on percentiles** → Ensures balanced categories
4. **Separate home/away features** → No asymmetry assumptions
5. **Derived features for model** → Pre-compute useful differentials

This dataframe structure gives you maximum flexibility for Bayesian Network modeling while maintaining interpretability and preventing data leakage!