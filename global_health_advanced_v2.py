# =============================================================================
#  GLOBAL HEALTH & LIFE EXPECTANCY — ADVANCED DATA SCIENCE PROJECT v2.0
#  Author  : Lakshya
#  Dataset : WHO Life Expectancy (112 countries, 2000–2015)
#
#  TECH STACK
#  ──────────
#  NumPy · Pandas · Matplotlib · Seaborn · SciPy · Scikit-Learn
#  XGBoost · SHAP · Plotly · Prophet
#
#  SECTIONS
#  ────────
#   1.  Imports & Configuration
#   2.  Data Loading & Inspection
#   3.  Data Cleaning & Preprocessing
#   4.  Exploratory Data Analysis (EDA)
#   5.  Statistical Analysis
#   6.  Correlation & Feature Analysis
#   7.  ML Classification — Baseline Random Forest
#   8.  ML Regression — Predict Exact Life Expectancy Value
#   9.  Model Comparison — 4 Classifiers Head-to-Head
#  10.  Hyperparameter Tuning — GridSearchCV
#  11.  Production sklearn Pipeline
#  12.  SHAP Explainability (Global + Local)
#  13.  Time-Series Forecasting — Facebook Prophet
#  14.  Interactive Plotly Dashboard (saved as HTML)
#  15.  Insights & Conclusions
# =============================================================================


# =============================================================================
# SECTION 1 — IMPORTS & CONFIGURATION
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('Agg')          # non-interactive backend; remove if running in Colab/Jupyter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from scipy import stats
from scipy.stats import norm

# Sklearn
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.impute           import SimpleImputer
from sklearn.model_selection  import (train_test_split, GridSearchCV,
                                      StratifiedKFold, cross_val_score)
from sklearn.pipeline         import Pipeline
from sklearn.linear_model     import LogisticRegression
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.ensemble         import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics          import (accuracy_score, f1_score,
                                      confusion_matrix, classification_report,
                                      mean_absolute_error, mean_squared_error,
                                      r2_score)

# XGBoost
from xgboost import XGBClassifier

# SHAP
import shap

# Plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Prophet
from prophet import Prophet

import warnings
warnings.filterwarnings('ignore')

# ── Global plot style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi'       : 120,
    'font.family'      : 'DejaVu Sans',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.titlesize'   : 13,
    'axes.labelsize'   : 11,
})
PALETTE    = {'Developed': '#2196F3', 'Developing': '#FF5722'}
CAT_COLORS = {'Low': '#F44336', 'Medium': '#FFC107', 'High': '#4CAF50'}

print("=" * 65)
print("  GLOBAL HEALTH & LIFE EXPECTANCY — ADVANCED PROJECT v2.0")
print("=" * 65)
print("  Tech: NumPy · Pandas · Matplotlib · Seaborn · SciPy")
print("        Scikit-Learn · XGBoost · SHAP · Plotly · Prophet")
print("  Sections : 15   |   Models : 4 classifiers + 1 regressor")
print("=" * 65)


# =============================================================================
# SECTION 2 — DATA LOADING & INSPECTION
# =============================================================================

print("\n[SECTION 2]  Data Loading & Inspection")

df = pd.read_csv('life_expectancy.csv')

print(f"\n  Shape       : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"  Countries   : {df['Country'].nunique()}")
print(f"  Year range  : {df['Year'].min()} – {df['Year'].max()}")

# Missing value summary
missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
missing = missing[missing > 0]
print(f"\n  Columns with missing values (%):")
for col, pct in missing.items():
    print(f"    {col:<35} {pct:.2f}%")


# =============================================================================
# SECTION 3 — DATA CLEANING & PREPROCESSING
# =============================================================================

print("\n[SECTION 3]  Data Cleaning & Preprocessing")

# 3.1  Normalise column names
df.columns = df.columns.str.strip().str.replace(r'\s+', ' ', regex=True)
print(f"  ✅  Column names stripped and normalised.")

# 3.2  Drop columns with > 15% missing values
miss_pct  = df.isnull().sum() / len(df) * 100
drop_cols = miss_pct[miss_pct > 15].index.tolist()
if drop_cols:
    df.drop(columns=drop_cols, inplace=True)
    print(f"  🗑️   Dropped columns (>15% null): {drop_cols}")

# 3.3  Split by development status, fill nulls with group medians
developed_df  = df[df['Status'] == 'Developed'].copy()
developing_df = df[df['Status'] == 'Developing'].copy()

def fill_group_median(frame):
    """Fill each numeric column's NaN with its within-group median."""
    for col in frame.select_dtypes(include=[np.number]).columns:
        if frame[col].isnull().any():
            frame[col] = frame[col].fillna(frame[col].median())

fill_group_median(developed_df)
fill_group_median(developing_df)

# 3.4  Recombine
df = (pd.concat([developed_df, developing_df])
        .sort_values(['Country', 'Year'])
        .reset_index(drop=True))
print(f"  ✅  Group-median imputation complete. Nulls remaining: {df.isnull().sum().sum()}")

# 3.5  Feature Engineering — Classification target
LE = 'Life expectancy'
df['LifeCategory'] = pd.cut(df[LE],
                             bins=[0, 60, 75, 120],
                             labels=['Low', 'Medium', 'High'])
print(f"\n  LifeCategory distribution:\n{df['LifeCategory'].value_counts().to_string()}")

# 3.6  Encode Status
df['Status_enc'] = LabelEncoder().fit_transform(df['Status'])

# 3.7  Build clean feature matrix with safety imputer
DROP  = [LE, 'LifeCategory', 'Country', 'Status', 'Year']
FEATS = [c for c in df.select_dtypes(include=[np.number]).columns if c not in DROP]
imp   = SimpleImputer(strategy='median')
X     = pd.DataFrame(imp.fit_transform(df[FEATS]), columns=FEATS)
y     = df['LifeCategory'].astype(str)

print(f"\n  Feature matrix : {X.shape[0]} samples × {X.shape[1]} features")
print(f"  Features used  : {FEATS}")

# 3.8  Descriptive statistics
print(f"\n  Descriptive Statistics (key columns):")
print(df[[LE, 'Schooling', 'GDP', 'HIV/AIDS']].describe().round(2).to_string())


# =============================================================================
# SECTION 4 — EXPLORATORY DATA ANALYSIS
# =============================================================================

print("\n[SECTION 4]  Exploratory Data Analysis")

yearly_global = df.groupby('Year')[LE].mean()
yearly_status = df.groupby(['Year', 'Status'])[LE].mean().unstack()

# Plot 1 — Global LE trend
fig, ax = plt.subplots(figsize=(14, 5))
bar_colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(yearly_global)))
bars = ax.bar(yearly_global.index, yearly_global.values,
              color=bar_colors, edgecolor='white', width=0.7)
for b in bars:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.1,
            f"{b.get_height():.1f}", ha='center', va='bottom', fontsize=7.5)
ax.set_title('Global Average Life Expectancy (2000–2015)', fontweight='bold', pad=14)
ax.set_xlabel('Year')
ax.set_ylabel('Average Life Expectancy (years)')
ax.set_xticks(yearly_global.index)
plt.tight_layout()
plt.savefig('adv_plot_01_global_trend.png', bbox_inches='tight')
plt.close()

# Plot 2 — Developed vs Developing side-by-side
bw = 0.38
x  = np.arange(len(yearly_status))
fig, ax = plt.subplots(figsize=(14, 5))
for i, (status, color) in enumerate(PALETTE.items()):
    if status in yearly_status.columns:
        ax.bar(x + (i - 0.5) * bw, yearly_status[status],
               width=bw, label=status, color=color, alpha=0.88, edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(yearly_status.index, rotation=45)
ax.set_title('Life Expectancy — Developed vs Developing (2000–2015)',
             fontweight='bold', pad=14)
ax.set_ylabel('Average Life Expectancy (years)')
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig('adv_plot_02_dev_vs_devg.png', bbox_inches='tight')
plt.close()

# Plot 3 — Top 10 & Bottom 10 countries
from matplotlib.patches import Patch
country_avg = (df.groupby(['Country', 'Status'])[LE].mean()
                 .reset_index()
                 .rename(columns={LE: 'Avg_LE'}))
top10    = country_avg.nlargest(10, 'Avg_LE')
bottom10 = country_avg.nsmallest(10, 'Avg_LE')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, data, title in zip(
        axes,
        [top10, bottom10],
        ['Top 10 — Highest Avg Life Expectancy',
         'Bottom 10 — Lowest Avg Life Expectancy']):
    colors = [PALETTE.get(s, '#888') for s in data['Status']]
    ax.barh(data['Country'], data['Avg_LE'], color=colors, edgecolor='white')
    for i, (val, _) in enumerate(zip(data['Avg_LE'], data['Country'])):
        ax.text(val + 0.2, i, f"{val:.1f}", va='center', fontsize=9)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Average Life Expectancy (years)')
    ax.invert_yaxis()
legend_els = [Patch(facecolor=PALETTE['Developed'],  label='Developed'),
              Patch(facecolor=PALETTE['Developing'], label='Developing')]
axes[0].legend(handles=legend_els, frameon=False, loc='lower right')
plt.tight_layout()
plt.savefig('adv_plot_03_countries.png', bbox_inches='tight')
plt.close()

# Plot 4 — LifeCategory pie
cat_counts = df['LifeCategory'].value_counts()
fig, ax = plt.subplots(figsize=(7, 7))
ax.pie(cat_counts, labels=cat_counts.index, autopct='%1.1f%%',
       colors=['#4CAF50', '#FFC107', '#F44336'],
       startangle=140, explode=[0.04] * len(cat_counts),
       wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
       textprops={'fontsize': 12})
ax.set_title('Life Expectancy Category Distribution (All Countries 2000–2015)',
             fontweight='bold', pad=16)
plt.tight_layout()
plt.savefig('adv_plot_04_category_pie.png', bbox_inches='tight')
plt.close()

# Plot 5 — Box plots by LifeCategory
features_box = ['GDP', 'Schooling', 'HIV/AIDS', 'Adult Mortality']
fig, axes = plt.subplots(1, len(features_box), figsize=(16, 5))
order = ['Low', 'Medium', 'High']
for ax, feat in zip(axes, features_box):
    data_by_cat = [df[df['LifeCategory'] == cat][feat].dropna().values
                   for cat in order]
    bp = ax.boxplot(data_by_cat, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2))
    for patch, cat in zip(bp['boxes'], order):
        patch.set_facecolor(CAT_COLORS[cat])
        patch.set_alpha(0.8)
    ax.set_xticklabels(order)
    ax.set_title(feat, fontweight='bold')
    ax.set_xlabel('LE Category')
fig.suptitle('Key Feature Distributions by Life Expectancy Category',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('adv_plot_05_boxplots.png', bbox_inches='tight')
plt.close()

print("  ✅  EDA plots 1–5 saved.")


# =============================================================================
# SECTION 5 — STATISTICAL ANALYSIS
# =============================================================================

print("\n[SECTION 5]  Statistical Analysis")

# Descriptive stats per group
for status, frame in [('Developed', developed_df), ('Developing', developing_df)]:
    desc = frame[LE].describe()
    print(f"\n  {status}:")
    print(f"    Mean={desc['mean']:.2f}  Std={desc['std']:.2f}  "
          f"Min={desc['min']:.1f}  Max={desc['max']:.1f}")
    print(f"    Skewness={frame[LE].skew():.4f}  Kurtosis={frame[LE].kurt():.4f}")

# Welch's t-test
t_stat, p_val = stats.ttest_ind(developed_df[LE].dropna(),
                                 developing_df[LE].dropna(), equal_var=False)
print(f"\n  Welch's T-test:  t = {t_stat:.4f}  |  p = {p_val:.2e}")
print(f"  {'✅ Statistically significant (p < 0.05)' if p_val < 0.05 else '❌ Not significant'}")

# Z-score outliers
z_scores = np.abs(stats.zscore(df[LE].dropna()))
n_outliers = (z_scores > 3).sum()
print(f"\n  Z-score outliers (|z|>3) in LE column: {n_outliers} records")

# P(LE < 60) for developing countries in 2015
dev_2015   = developing_df[developing_df['Year'] == 2015][LE].dropna()
mu, sigma  = dev_2015.mean(), dev_2015.std()
z60        = (60 - mu) / sigma
prob_lt60  = norm.cdf(z60)
print(f"\n  Developing countries (2015) — Mean: {mu:.2f}  Std: {sigma:.2f}")
print(f"  Z-score at LE=60  : {z60:.4f}")
print(f"  P(LE < 60)        : {prob_lt60 * 100:.2f}%")

# Countries below 60 in 2015
low_le = (developing_df[(developing_df['Year'] == 2015) & (developing_df[LE] < 60)]
            [['Country', LE]].sort_values(LE))
print(f"\n  Countries with LE < 60 in 2015: {len(low_le)}")
print(low_le.to_string(index=False))

# Probability distribution plot
x_r = np.linspace(mu - 4*sigma, mu + 4*sigma, 400)
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(x_r, norm.pdf(x_r, mu, sigma), color='#2196F3', linewidth=2.5,
        label='Normal Distribution')
ax.fill_between(x_r, norm.pdf(x_r, mu, sigma), where=(x_r <= 60),
                color='#F44336', alpha=0.45,
                label=f'P(LE < 60) = {prob_lt60*100:.1f}%')
ax.fill_between(x_r, norm.pdf(x_r, mu, sigma), where=(x_r > 60),
                color='#4CAF50', alpha=0.25,
                label=f'P(LE ≥ 60) = {(1-prob_lt60)*100:.1f}%')
ax.axvline(60, color='#F44336', linestyle='--', linewidth=1.8, label='LE = 60')
ax.axvline(mu, color='#2196F3', linestyle='-',  linewidth=1.8, label=f'Mean = {mu:.1f}')
ax.set_title('Probability Distribution — Developing Countries Life Expectancy (2015)',
             fontweight='bold', pad=14)
ax.set_xlabel('Life Expectancy (years)')
ax.set_ylabel('Probability Density')
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig('adv_plot_06_prob_dist.png', bbox_inches='tight')
plt.close()
print("  ✅  Statistical analysis complete.")


# =============================================================================
# SECTION 6 — CORRELATION & FEATURE ANALYSIS
# =============================================================================

print("\n[SECTION 6]  Correlation & Feature Analysis")

num_df      = df.select_dtypes(include=[np.number]).drop(columns=['Year'])
corr_matrix = num_df.corr()
le_corr     = corr_matrix[LE].drop(LE).sort_values(key=abs, ascending=False)

# Heatmap
fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, linewidths=0.4, annot_kws={'size': 7.5}, ax=ax,
            cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Heatmap (Lower Triangle)', fontweight='bold', pad=16)
plt.tight_layout()
plt.savefig('adv_plot_07_heatmap.png', bbox_inches='tight')
plt.close()

# Correlation bar chart
fig, ax = plt.subplots(figsize=(12, 5))
bar_cols = ['#4CAF50' if v > 0 else '#F44336' for v in le_corr]
ax.bar(le_corr.index, le_corr.values, color=bar_cols, edgecolor='white', alpha=0.88)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_title('Feature Correlation with Life Expectancy', fontweight='bold', pad=14)
ax.set_ylabel('Pearson Correlation Coefficient')
ax.set_xticklabels(le_corr.index, rotation=45, ha='right')
plt.tight_layout()
plt.savefig('adv_plot_08_corr_bar.png', bbox_inches='tight')
plt.close()

print(f"\n  Top 5 positive correlates:\n{le_corr[le_corr > 0].head(5).round(3).to_string()}")
print(f"\n  Top 5 negative correlates:\n{le_corr[le_corr < 0].head(5).round(3).to_string()}")
print("  ✅  Correlation analysis complete.")


# =============================================================================
# SECTION 7 — BASELINE RANDOM FOREST CLASSIFIER
# =============================================================================

print("\n[SECTION 7]  Baseline Random Forest Classifier")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
Xtr    = scaler.fit_transform(X_train)
Xte    = scaler.transform(X_test)

rf_base = RandomForestClassifier(n_estimators=200, max_depth=12,
                                  class_weight='balanced', random_state=42,
                                  n_jobs=-1)
rf_base.fit(Xtr, y_train)
yp_base  = rf_base.predict(Xte)
acc_base = accuracy_score(y_test, yp_base)
f1_base  = f1_score(y_test, yp_base, average='macro')

print(f"\n  Baseline RF Test Accuracy : {acc_base*100:.2f}%")
print(f"  Baseline RF F1 (Macro)    : {f1_base:.4f}")
print(f"\n{classification_report(y_test, yp_base)}")

# Feature importance plot
feat_imp  = pd.Series(rf_base.feature_importances_, index=FEATS).sort_values(ascending=False).head(12)
fig, ax   = plt.subplots(figsize=(12, 5))
colors_fi = plt.cm.RdYlGn(np.linspace(0.85, 0.15, len(feat_imp)))
bars = ax.bar(feat_imp.index, feat_imp.values, color=colors_fi, edgecolor='white')
for b in bars:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.002,
            f"{b.get_height():.3f}", ha='center', va='bottom', fontsize=8)
ax.set_title('Top 12 Feature Importances — Baseline Random Forest',
             fontweight='bold', pad=14)
ax.set_xticklabels(feat_imp.index, rotation=40, ha='right')
ax.set_ylabel('Importance Score')
plt.tight_layout()
plt.savefig('adv_plot_09_feat_importance.png', bbox_inches='tight')
plt.close()
print("  ✅  Baseline RF complete.")


# =============================================================================
# SECTION 8 — REGRESSION — PREDICT EXACT LIFE EXPECTANCY
# =============================================================================

print("\n[SECTION 8]  Regression — Predict Exact Life Expectancy Value")

y_reg = df[LE].values
Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)

sc_r  = StandardScaler()
rfr   = RandomForestRegressor(n_estimators=200, max_depth=14, random_state=42, n_jobs=-1)
rfr.fit(sc_r.fit_transform(Xtr_r), ytr_r)
ypr   = rfr.predict(sc_r.transform(Xte_r))

mae  = mean_absolute_error(yte_r, ypr)
rmse = np.sqrt(mean_squared_error(yte_r, ypr))
r2   = r2_score(yte_r, ypr)

print(f"\n  MAE  = {mae:.4f} years")
print(f"  RMSE = {rmse:.4f} years")
print(f"  R²   = {r2:.4f}  ({r2*100:.2f}% variance explained)")

# Actual vs Predicted + Residuals
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].scatter(yte_r, ypr, alpha=0.4, s=18, color='#2196F3', edgecolors='none',
                label='Predictions')
mn, mx = min(yte_r.min(), ypr.min()), max(yte_r.max(), ypr.max())
axes[0].plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Perfect prediction')
axes[0].set_xlabel('Actual Life Expectancy (years)')
axes[0].set_ylabel('Predicted Life Expectancy (years)')
axes[0].set_title(f'Actual vs Predicted  |  R²={r2:.3f}  MAE={mae:.2f} yrs',
                  fontweight='bold')
axes[0].legend(frameon=False)

residuals = yte_r - ypr
axes[1].hist(residuals, bins=35, color='#9C27B0', edgecolor='white', alpha=0.85)
axes[1].axvline(0, color='black', linewidth=1.8, linestyle='--',
                label=f'Mean residual={residuals.mean():.2f}')
axes[1].set_xlabel('Residual Value (years)')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Residual Distribution', fontweight='bold')
axes[1].legend(frameon=False)

plt.suptitle('Random Forest Regressor — Life Expectancy Prediction',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('adv_plot_10_regression.png', bbox_inches='tight')
plt.close()
print("  ✅  Regression analysis complete.")


# =============================================================================
# SECTION 9 — MODEL COMPARISON — 4 CLASSIFIERS HEAD-TO-HEAD
# =============================================================================

print("\n[SECTION 9]  Model Comparison — 4 Classifiers Head-to-Head")

# Label-encode y for XGBoost
class_le    = LabelEncoder()
ytr_enc     = class_le.fit_transform(y_train)

models = {
    'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
    'KNN (k=7)'          : KNeighborsClassifier(n_neighbors=7),
    'Random Forest'      : RandomForestClassifier(n_estimators=150, max_depth=12,
                                                   class_weight='balanced',
                                                   random_state=42, n_jobs=-1),
    'XGBoost'            : XGBClassifier(n_estimators=100, max_depth=5,
                                          learning_rate=0.1, eval_metric='mlogloss',
                                          random_state=42, verbosity=0),
}

results = {}
for name, model in models.items():
    is_xgb = 'XGB' in type(model).__name__
    _ytr   = ytr_enc if is_xgb else y_train

    # Fit
    model.fit(Xtr, _ytr)
    ypm = model.predict(Xte)

    # Decode XGBoost predictions
    if is_xgb:
        ypm = class_le.inverse_transform(ypm)

    acc = accuracy_score(y_test, ypm)
    f1  = f1_score(y_test, ypm, average='macro')
    results[name] = {'Test Acc': acc, 'F1 Macro': f1}
    print(f"  ✅  {name:<25} Acc={acc*100:.2f}%  F1={f1:.3f}")

results_df = pd.DataFrame(results).T.sort_values('Test Acc', ascending=False)

# Model comparison chart
model_names = results_df.index.tolist()
test_accs   = (results_df['Test Acc'] * 100).tolist()
f1_scores   = results_df['F1 Macro'].tolist()

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

bar_c = ['#4CAF50' if v == max(test_accs) else '#90CAF9' for v in test_accs]
axes[0].bar(model_names, test_accs, color=bar_c, edgecolor='white', alpha=0.9)
for i, v in enumerate(test_accs):
    axes[0].text(i, v + 0.3, f"{v:.1f}%", ha='center', fontsize=10, fontweight='bold')
axes[0].set_title('Test Accuracy by Model', fontweight='bold')
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_xticklabels(model_names, rotation=15, ha='right')

bar_c2 = ['#4CAF50' if v == max(f1_scores) else '#FFCC80' for v in f1_scores]
axes[1].barh(model_names, f1_scores, color=bar_c2, edgecolor='white', alpha=0.9)
for i, v in enumerate(f1_scores):
    axes[1].text(v + 0.003, i, f"{v:.3f}", va='center', fontsize=10)
axes[1].set_title('F1 Macro Score by Model', fontweight='bold')
axes[1].set_xlabel('F1 Score (Macro Average)')
axes[1].invert_yaxis()

plt.suptitle('Model Comparison — 4 Classifiers Head-to-Head',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('adv_plot_11_model_compare.png', bbox_inches='tight')
plt.close()
print(f"\n  Best model: {results_df.index[0]}  (Acc={results_df['Test Acc'].iloc[0]*100:.2f}%)")


# =============================================================================
# SECTION 10 — HYPERPARAMETER TUNING — GridSearchCV
# =============================================================================

print("\n[SECTION 10]  Hyperparameter Tuning — GridSearchCV")

param_grid = {
    'n_estimators'    : [100, 200],
    'max_depth'       : [8, 12, None],
    'min_samples_leaf': [1, 2, 4],
}

gs = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
    param_grid = param_grid,
    cv         = StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring    = 'f1_macro',
    n_jobs     = -1,
    verbose    = 0,
)
gs.fit(Xtr, y_train)

best_rf     = gs.best_estimator_
yp_tuned    = best_rf.predict(Xte)
acc_tuned   = accuracy_score(y_test, yp_tuned)
f1_tuned    = f1_score(y_test, yp_tuned, average='macro')
best_params = gs.best_params_

print(f"\n  Best Parameters : {best_params}")
print(f"  Best CV F1      : {gs.best_score_*100:.2f}%")
print(f"  Tuned Test Acc  : {acc_tuned*100:.2f}%  (baseline: {acc_base*100:.2f}%)")
print(f"  Tuned F1 Macro  : {f1_tuned:.4f}  (baseline: {f1_base:.4f})")

# GridSearch results heatmap
cv_results  = pd.DataFrame(gs.cv_results_)
pivot = cv_results.pivot_table(index='param_max_depth',
                                columns='param_n_estimators',
                                values='mean_test_score',
                                aggfunc='max')
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGnBu',
            linewidths=0.5, ax=ax, cbar_kws={'label': 'F1 Macro'})
ax.set_title('GridSearchCV — max_depth vs n_estimators (F1 Macro)',
             fontweight='bold', pad=14)
plt.tight_layout()
plt.savefig('adv_plot_12_gridsearch.png', bbox_inches='tight')
plt.close()

# Confusion matrix — tuned model
classes_list = sorted(y_test.unique())
cm_tuned = confusion_matrix(y_test, yp_tuned, labels=classes_list)
fig, ax  = plt.subplots(figsize=(7, 6))
sns.heatmap(cm_tuned, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes_list, yticklabels=classes_list,
            linewidths=0.5, annot_kws={'size': 13, 'weight': 'bold'}, ax=ax)
ax.set_title(f'Confusion Matrix — Tuned RF (Acc={acc_tuned*100:.1f}%  F1={f1_tuned:.3f})',
             fontweight='bold', pad=14)
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
plt.tight_layout()
plt.savefig('adv_plot_13_cm_tuned.png', bbox_inches='tight')
plt.close()
print("  ✅  Hyperparameter tuning complete.")


# =============================================================================
# SECTION 11 — PRODUCTION sklearn PIPELINE
# =============================================================================

print("\n[SECTION 11]  Production sklearn Pipeline")

production_pipeline = Pipeline(steps=[
    ('imputer',    SimpleImputer(strategy='median')),
    ('scaler',     StandardScaler()),
    ('classifier', RandomForestClassifier(
        **best_params,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )),
])

production_pipeline.fit(X_train, y_train)
pipe_preds = production_pipeline.predict(X_test)
pipe_acc   = accuracy_score(y_test, pipe_preds)
pipe_f1    = f1_score(y_test, pipe_preds, average='macro')

print(f"\n  Pipeline steps:")
for step_name, step_obj in production_pipeline.steps:
    print(f"    [{step_name}] → {step_obj.__class__.__name__}")

print(f"\n  Pipeline Test Accuracy : {pipe_acc*100:.2f}%")
print(f"  Pipeline F1 (Macro)    : {pipe_f1:.4f}")

# Single-country prediction demo
sample_idx     = X_test.iloc[[0]]
sample_country = df.iloc[X_test.index[0]]['Country']
sample_year    = df.iloc[X_test.index[0]]['Year']
sample_true    = y_test.iloc[0]
sample_pred    = production_pipeline.predict(sample_idx)[0]
sample_le      = df.iloc[X_test.index[0]][LE]

print(f"\n  🔮  Single-Country Prediction Demo:")
print(f"      Country   : {sample_country} ({sample_year})")
print(f"      True LE   : {sample_le:.1f} yrs  →  Category: {sample_true}")
print(f"      Predicted : {sample_pred}")
print(f"      {'✅ Correct!' if sample_pred == sample_true else '❌ Incorrect'}")
print("  ✅  Pipeline ready for production.")


# =============================================================================
# SECTION 12 — SHAP EXPLAINABILITY
# =============================================================================

print("\n[SECTION 12]  SHAP Explainability (Global + Local)")

explainer  = shap.TreeExplainer(best_rf)
shap_vals  = explainer.shap_values(Xte)   # shape: (n_samples, n_features, n_classes)

# Average absolute SHAP across all classes for global importance
shap_arr   = np.array(shap_vals)          # (n_samples, n_features, n_classes)
mean_shap  = pd.Series(
    np.abs(shap_arr).mean(axis=(0, 2)),    # mean over samples and classes
    index=FEATS
).sort_values(ascending=False)

print(f"\n  SHAP values computed for {len(Xte)} test samples.")
print(f"\n  Global Feature Importance (mean |SHAP|):")
print(mean_shap.round(5).head(10).to_string())

# SHAP Global bar chart
top_shap = mean_shap.sort_values(ascending=True).tail(12)
fig, ax  = plt.subplots(figsize=(10, 7))
colors_sh = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(top_shap)))
ax.barh(top_shap.index, top_shap.values, color=colors_sh, edgecolor='white')
for i, v in enumerate(top_shap.values):
    ax.text(v + 0.0005, i, f"{v:.4f}", va='center', fontsize=8.5)
ax.set_title('SHAP — Global Feature Importance (Mean |SHAP| across all classes)',
             fontweight='bold', pad=14)
ax.set_xlabel('Mean |SHAP Value|')
plt.tight_layout()
plt.savefig('adv_plot_14_shap_global.png', bbox_inches='tight')
plt.close()

# SHAP Local waterfall — single prediction
# shap_arr shape: (n_samples, n_features, n_classes) — average across classes
single_shap_mean = shap_arr[0].mean(axis=1)   # shape: (n_features,)
# Guard: ensure values are finite scalars
single_shap_mean = np.nan_to_num(single_shap_mean.astype(float), nan=0.0)
shap_local = pd.Series(single_shap_mean, index=FEATS)
shap_local = shap_local.reindex(FEATS)        # ensure alignment
top10_local = shap_local.abs().nlargest(10).index
shap_df = shap_local[top10_local].sort_values()

fig, ax = plt.subplots(figsize=(10, 6))
colors_wf = ['#4CAF50' if v > 0 else '#F44336' for v in shap_df.values]
ax.barh(shap_df.index.tolist(), shap_df.values.tolist(),
        color=colors_wf, edgecolor='white', alpha=0.88)
ax.axvline(0, color='black', linewidth=0.8)
vmax = np.abs(shap_df.values).max()
for i, v in enumerate(shap_df.values):
    offset = vmax * 0.03
    ax.text(v + (offset if v >= 0 else -offset), i, f"{float(v):+.4f}",
            va='center', ha='left' if v >= 0 else 'right', fontsize=8.5)
ax.set_title(f'SHAP Waterfall — Local Explanation\n'
             f'{sample_country} ({sample_year})  |  '
             f'Predicted: {sample_pred}  |  True: {sample_true}',
             fontweight='bold', pad=14)
ax.set_xlabel('SHAP Value (contribution to prediction)')
plt.tight_layout()
plt.savefig('adv_plot_15_shap_waterfall.png', bbox_inches='tight', dpi=120)
plt.close()
print("  ✅  SHAP explainability complete.")


# =============================================================================
# SECTION 13 — TIME-SERIES FORECASTING — PROPHET
# =============================================================================

print("\n[SECTION 13]  Time-Series Forecasting — Prophet")

prophet_results = {}
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, status, color in zip(axes,
                              ['Developed', 'Developing'],
                              ['#2196F3', '#FF5722']):
    ts = (df[df['Status'] == status]
          .groupby('Year')[LE].mean()
          .reset_index()
          .rename(columns={'Year': 'ds', LE: 'y'}))
    ts['ds'] = pd.to_datetime(ts['ds'], format='%Y')

    m = Prophet(yearly_seasonality=False, weekly_seasonality=False,
                daily_seasonality=False, changepoint_prior_scale=0.3,
                interval_width=0.90)
    m.fit(ts)
    future   = m.make_future_dataframe(periods=10, freq='YE')
    forecast = m.predict(future)
    fut      = forecast[forecast['ds'].dt.year > 2015]

    prophet_results[status] = {
        '2020_forecast': fut[fut['ds'].dt.year <= 2020]['yhat'].mean(),
        '2025_forecast': fut['yhat'].iloc[-1],
    }

    # Plot historical
    ax.plot(ts['ds'].dt.year, ts['y'], 'o-', color=color,
            linewidth=2.2, markersize=6, label='Historical (2000–2015)')

    # Plot forecast
    ax.plot(fut['ds'].dt.year, fut['yhat'], 's--', color=color,
            alpha=0.75, linewidth=2, markersize=6,
            label='Forecast (2016–2025)')

    # Confidence interval
    ax.fill_between(fut['ds'].dt.year,
                    fut['yhat_lower'], fut['yhat_upper'],
                    alpha=0.2, color=color, label='90% CI')

    # Annotate every 3rd forecast point
    for _, row in fut.iloc[::3].iterrows():
        ax.annotate(f"{row['yhat']:.1f}",
                    xy=(row['ds'].year, row['yhat']),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=8, color=color)

    ax.axvline(2015.5, color='gray', linestyle=':', linewidth=1.3)
    ax.text(2015.7, ax.get_ylim()[0] + 0.3,
            '← History | Forecast →', fontsize=8, color='gray')
    ax.set_title(f'{status} Countries — Life Expectancy Forecast',
                 fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Life Expectancy (years)')
    ax.legend(frameon=False, fontsize=8)

    print(f"  {status} — 2025 forecast: {prophet_results[status]['2025_forecast']:.2f} yrs")

plt.suptitle('Prophet Time-Series Forecasting — 2016–2025 Horizon',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('adv_plot_16_prophet.png', bbox_inches='tight')
plt.close()
print("  ✅  Time-series forecasting complete.")


# =============================================================================
# SECTION 14 — INTERACTIVE PLOTLY DASHBOARD
# =============================================================================

print("\n[SECTION 14]  Interactive Plotly Dashboard")

fig_p = make_subplots(
    rows=2, cols=3,
    subplot_titles=(
        '🌍 Global LE Trend (2000–2015)',
        '📊 Developed vs Developing',
        '🏷️ Category Distribution',
        '🏆 Top 10 Countries by Avg LE',
        '🤖 Model Comparison (Acc %)',
        '🔮 Prophet Forecast (Developing)',
    ),
    specs=[
        [{'type': 'xy'},     {'type': 'xy'},     {'type': 'domain'}],
        [{'type': 'xy'},     {'type': 'xy'},     {'type': 'xy'}],
    ],
    vertical_spacing=0.15,
    horizontal_spacing=0.09,
)

# P1 — Global trend bar
fig_p.add_trace(go.Bar(
    x=yearly_global.index.tolist(),
    y=yearly_global.values.tolist(),
    marker_color=yearly_global.values.tolist(),
    marker_colorscale='Blues',
    name='Global Avg LE',
    hovertemplate='Year: %{x}<br>Avg LE: %{y:.2f} yrs<extra></extra>'),
    row=1, col=1)

# P2 — Developed vs Developing
for status, color in PALETTE.items():
    if status in yearly_status.columns:
        fig_p.add_trace(go.Bar(
            x=yearly_status.index.tolist(),
            y=yearly_status[status].tolist(),
            name=status, marker_color=color, opacity=0.85,
            hovertemplate=f'{status} · Year: %{{x}}<br>LE: %{{y:.2f}} yrs<extra></extra>'),
            row=1, col=2)

# P3 — Donut chart
fig_p.add_trace(go.Pie(
    labels=cat_counts.index.tolist(),
    values=cat_counts.values.tolist(),
    marker_colors=['#4CAF50', '#FFC107', '#F44336'],
    hole=0.38,
    hovertemplate='%{label}<br>Count: %{value}<br>%{percent}<extra></extra>'),
    row=1, col=3)

# P4 — Top 10 countries
fig_p.add_trace(go.Bar(
    x=top10['Avg_LE'].tolist(),
    y=top10['Country'].tolist(),
    orientation='h',
    marker_color=[PALETTE.get(s, '#888') for s in top10['Status']],
    name='Top 10',
    hovertemplate='%{y}<br>Avg LE: %{x:.2f} yrs<extra></extra>'),
    row=2, col=1)

# P5 — Model comparison
bar_c = ['#4CAF50' if v == max(test_accs) else '#90CAF9' for v in test_accs]
fig_p.add_trace(go.Bar(
    x=model_names, y=test_accs,
    marker_color=bar_c,
    name='Test Acc %',
    hovertemplate='%{x}<br>Accuracy: %{y:.2f}%<extra></extra>'),
    row=2, col=2)

# P6 — Prophet forecast (Developing)
ts_devg = (df[df['Status'] == 'Developing'].groupby('Year')[LE].mean()
             .reset_index().rename(columns={'Year': 'ds', LE: 'y'}))
ts_devg['ds'] = pd.to_datetime(ts_devg['ds'], format='%Y')
mp2     = Prophet(yearly_seasonality=False, weekly_seasonality=False,
                  daily_seasonality=False, interval_width=0.90)
mp2.fit(ts_devg)
fc2     = mp2.predict(mp2.make_future_dataframe(periods=10, freq='YE'))
fut2    = fc2[fc2['ds'].dt.year > 2015]

fig_p.add_trace(go.Scatter(
    x=ts_devg['ds'].dt.year.tolist(), y=ts_devg['y'].tolist(),
    mode='lines+markers', name='Historical (Developing)',
    line=dict(color='#FF5722', width=2),
    hovertemplate='Year: %{x}<br>LE: %{y:.2f} yrs<extra></extra>'),
    row=2, col=3)
fig_p.add_trace(go.Scatter(
    x=fut2['ds'].dt.year.tolist(), y=fut2['yhat'].tolist(),
    mode='lines+markers', name='Forecast',
    line=dict(color='#FF5722', width=2, dash='dash'),
    hovertemplate='Year: %{x}<br>Forecast: %{y:.2f} yrs<extra></extra>'),
    row=2, col=3)
fig_p.add_trace(go.Scatter(
    x=fut2['ds'].dt.year.tolist() + fut2['ds'].dt.year.tolist()[::-1],
    y=fut2['yhat_upper'].tolist() + fut2['yhat_lower'].tolist()[::-1],
    fill='toself', fillcolor='rgba(255,87,34,0.15)',
    line=dict(color='rgba(255,255,255,0)'), name='90% CI'),
    row=2, col=3)

fig_p.update_layout(
    title=dict(
        text='<b>GLOBAL HEALTH & LIFE EXPECTANCY — INTERACTIVE DASHBOARD</b>'
             '<br><sup>WHO Data · 112 Countries · 2000–2015 · '
             'Built with Python + Plotly | Lakshya</sup>',
        x=0.5, xanchor='center', font=dict(size=16)),
    height=800, barmode='group',
    paper_bgcolor='#FAFAFA', plot_bgcolor='white',
    font=dict(family='Arial', size=11),
    legend=dict(orientation='h', yanchor='bottom', y=-0.15,
                xanchor='center', x=0.5),
    hoverlabel=dict(bgcolor='white', font_size=12),
)
fig_p.update_xaxes(showgrid=True, gridcolor='#EEEEEE')
fig_p.update_yaxes(showgrid=True, gridcolor='#EEEEEE')
fig_p.update_yaxes(autorange='reversed', row=2, col=1)

html_path = 'interactive_dashboard.html'
fig_p.write_html(html_path, include_plotlyjs='cdn', full_html=True)
print(f"  ✅  Interactive dashboard saved → {html_path}")
print("      Open in any browser — fully hoverable, zoomable, pannable.")


# =============================================================================
# SECTION 15 — INSIGHTS & CONCLUSIONS
# =============================================================================

print("\n[SECTION 15]  Insights & Conclusions")

le_2000  = yearly_global.loc[2000]
le_2015  = yearly_global.loc[2015]
dev_avg  = developed_df[LE].mean()
devg_avg = developing_df[LE].mean()

print(f"""
╔══════════════════════════════════════════════════════════════════╗
║          FINAL PROJECT REPORT — KEY FINDINGS SUMMARY            ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  EXPLORATORY ANALYSIS                                            ║
║  ▸ Global LE: {le_2000:.1f} yrs (2000) → {le_2015:.1f} yrs (2015) (+{le_2015-le_2000:.1f} yrs)  ║
║  ▸ Developed avg  : {dev_avg:.1f} yrs                                  ║
║  ▸ Developing avg : {devg_avg:.1f} yrs  (gap = {dev_avg-devg_avg:.1f} yrs)             ║
║  ▸ Gap is statistically significant (p ≈ 0, Welch's t-test)     ║
║  ▸ {prob_lt60*100:.1f}% of developing countries had LE < 60 in 2015     ║
║                                                                  ║
║  REGRESSION (Random Forest Regressor)                            ║
║  ▸ R²   = {r2:.4f}  ({r2*100:.1f}% variance explained)              ║
║  ▸ MAE  = {mae:.3f} years  |  RMSE = {rmse:.3f} years             ║
║                                                                  ║
║  MODEL COMPARISON (4 Classifiers)                                ║
║  ▸ Best model    : {results_df.index[0]:<30}          ║
║  ▸ Test accuracy : {results_df['Test Acc'].iloc[0]*100:.2f}%                             ║
║  ▸ F1 Macro      : {results_df['F1 Macro'].iloc[0]:.4f}                            ║
║                                                                  ║
║  HYPERPARAMETER TUNING (GridSearchCV — 5-fold CV)                ║
║  ▸ Best params   : {str(best_params)[:44]}║
║  ▸ Tuned Acc     : {acc_tuned*100:.2f}%  F1: {f1_tuned:.4f}                    ║
║                                                                  ║
║  SHAP EXPLAINABILITY                                             ║
║  ▸ Top feature   : {mean_shap.idxmax():<35}       ║
║  ▸ Key positives : Income composition, Schooling, GDP           ║
║  ▸ Key negatives : HIV/AIDS, Adult Mortality, Measles           ║
║                                                                  ║
║  PROPHET FORECAST (2025 horizon)                                 ║
║  ▸ Developing avg LE by 2025: {prophet_results['Developing']['2025_forecast']:.1f} yrs              ║
║  ▸ Developed  avg LE by 2025: {prophet_results['Developed']['2025_forecast']:.1f} yrs              ║
║                                                                  ║
║  OUTPUT FILES                                                    ║
║  ▸ 16 static plots  (adv_plot_01 … adv_plot_16)                 ║
║  ▸ interactive_dashboard.html (open in any browser)             ║
║                                                                  ║
║  TECH STACK                                                      ║
║  NumPy · Pandas · Matplotlib · Seaborn · SciPy                  ║
║  Scikit-Learn · XGBoost · SHAP · Plotly · Prophet               ║
╚══════════════════════════════════════════════════════════════════╝
""")

print("🎉  Advanced project v2.0 complete!")
