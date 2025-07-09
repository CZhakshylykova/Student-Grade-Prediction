import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from contextlib import redirect_stdout
import warnings
import os

warnings.filterwarnings('ignore')

# ================================
# Ensure Plots Folder Exists
# ================================
if not os.path.exists('plots'):
    os.makedirs('plots')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load the dataset (change file as needed)
df = pd.read_csv('student-mat.csv', sep=';')   # or 'student-por.csv'

# ================================
# VISUALIZATION FUNCTIONS
# ================================

def create_comprehensive_plots():
    """Create comprehensive visualizations for the student performance dataset"""
    plt.rcParams['figure.figsize'] = (15, 10)

    # 1. Target Distribution: Final Grade (G3)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Student Performance - Exploratory Data Analysis', fontsize=16, fontweight='bold')

    # G3 distribution
    axes[0, 0].hist(df['G3'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Distribution of Final Grades (G3)')
    axes[0, 0].set_xlabel('Final Grade (G3)')
    axes[0, 0].set_ylabel('Count')

    # G1 vs G3 scatter
    axes[0, 1].scatter(df['G1'], df['G3'], alpha=0.5)
    axes[0, 1].set_title('G1 vs G3')
    axes[0, 1].set_xlabel('G1 (1st period grade)')
    axes[0, 1].set_ylabel('G3 (final grade)')

    # G2 vs G3 scatter
    axes[0, 2].scatter(df['G2'], df['G3'], alpha=0.5, color='orange')
    axes[0, 2].set_title('G2 vs G3')
    axes[0, 2].set_xlabel('G2 (2nd period grade)')
    axes[0, 2].set_ylabel('G3 (final grade)')

    # Study time
    axes[1, 0].boxplot([df[df['G3']>=10]['studytime'], df[df['G3']<10]['studytime']], labels=['Passed', 'Failed'])
    axes[1, 0].set_title('Study Time by Pass/Fail (G3>=10)')
    axes[1, 0].set_ylabel('Study Time')

    # Absences distribution
    axes[1, 1].hist(df['absences'], bins=20, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].set_title('Absences Distribution')
    axes[1, 1].set_xlabel('Absences')
    axes[1, 1].set_ylabel('Count')

    # Age distribution
    axes[1, 2].hist(df['age'], bins=8, alpha=0.7, color='lime', edgecolor='black')
    axes[1, 2].set_title('Age Distribution')
    axes[1, 2].set_xlabel('Age')
    axes[1, 2].set_ylabel('Count')

    plt.tight_layout()
    plt.savefig('plots/eda_main_grid.png')
    plt.close()

    # 2. Correlation Analysis
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.select_dtypes(include=[np.number]).corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix of Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/correlation_matrix.png')
    plt.close()

    # 3. Key Feature Distributions by Outcome (Pass/Fail)
    df['pass'] = (df['G3'] >= 10).astype(int)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Key Features Distribution by Pass/Fail', fontsize=14, fontweight='bold')

    for outcome, label in zip([1, 0], ['Pass', 'Fail']):
        axes[0, 0].hist(df[df['pass'] == outcome]['studytime'], alpha=0.7, label=label, bins=4)
    axes[0, 0].set_title('Study Time')
    axes[0, 0].legend()

    for outcome, label in zip([1, 0], ['Pass', 'Fail']):
        axes[0, 1].hist(df[df['pass'] == outcome]['absences'], alpha=0.7, label=label, bins=8)
    axes[0, 1].set_title('Absences')
    axes[0, 1].legend()

    for outcome, label in zip([1, 0], ['Pass', 'Fail']):
        axes[1, 0].hist(df[df['pass'] == outcome]['age'], alpha=0.7, label=label, bins=6)
    axes[1, 0].set_title('Age')
    axes[1, 0].legend()

    for outcome, label in zip([1, 0], ['Pass', 'Fail']):
        axes[1, 1].hist(df[df['pass'] == outcome]['failures'], alpha=0.7, label=label, bins=4)
    axes[1, 1].set_title('Past Failures')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('plots/features_by_pass_fail.png')
    plt.close()

    # 4. Categorical features analysis (Sex, School, Support, Family size)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Categorical Features Analysis', fontsize=14, fontweight='bold')

    categorical_features = ['sex', 'school', 'famsize', 'schoolsup', 'higher']

    for i, feature in enumerate(categorical_features):
        row = i // 3
        col = i % 3
        ct = pd.crosstab(df[feature], df['pass'])
        ct.plot(kind='bar', ax=axes[row, col])
        axes[row, col].set_title(f'{feature.title()} vs Pass/Fail')
        axes[row, col].set_xlabel(feature.title())
        axes[row, col].set_ylabel('Count')
        axes[row, col].legend(['Fail', 'Pass'])
        axes[row, col].tick_params(axis='x', rotation=0)

    fig.delaxes(axes[1, 2])
    plt.tight_layout()
    plt.savefig('plots/categorical_features.png')
    plt.close()

# ================================
# 7. STATISTICAL ANALYSIS
# ================================
def perform_statistical_tests():
    """Perform statistical tests to identify significant features"""
    print("Statistical Tests for Feature Significance:")
    print("=" * 50)
    # Numerical features - t-test (Pass/Fail)
    numerical_features = ['age', 'absences', 'G1', 'G2', 'failures', 'studytime']
    print("\nT-tests for Numerical Features:")
    for feature in numerical_features:
        if feature in df.columns:
            passed = df[df['pass'] == 1][feature]
            failed = df[df['pass'] == 0][feature]
            t_stat, p_value = stats.ttest_ind(passed, failed)
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"{feature}: t-statistic = {t_stat:.3f}, p-value = {p_value:.6f} {significance}")
    # Categorical features - Chi-square test
    print("\nChi-square Tests for Categorical Features:")
    categorical_features = ['sex', 'school', 'famsize', 'schoolsup', 'higher']
    for feature in categorical_features:
        if feature in df.columns:
            contingency_table = pd.crosstab(df[feature], df['pass'])
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"{feature}: Chi-square = {chi2:.3f}, p-value = {p_value:.6f} {significance}")
    print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05")

# ================================
# 8. ADVANCED ANALYSIS
# ================================
def advanced_analysis():
    """Perform advanced analysis including PCA and clustering"""
    numerical_cols = df.select_dtypes(include=[np.number]).columns.drop('G3')
    X = df[numerical_cols].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    print("Principal Component Analysis:")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_[:5]}")
    print(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)[:5]}")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
             np.cumsum(pca.explained_variance_ratio_), 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA - Cumulative Explained Variance')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    colors = ['red' if x == 0 else 'blue' for x in df['pass']]
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.6)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA - First Two Components')
    plt.legend(['Fail', 'Pass'])
    plt.tight_layout()
    plt.savefig('plots/pca_analysis.png')
    plt.close()
    feature_importance = pd.DataFrame({
        'Feature': numerical_cols,
        'PC1': abs(pca.components_[0]),
        'PC2': abs(pca.components_[1])
    }).sort_values('PC1', ascending=False)
    print("\nFeature Importance in First Principal Component:")
    print(feature_importance)

# ================================
# 9. GRADE GROUP ANALYSIS
# ================================
def grade_group_analysis():
    """Analyze performance by grade bands"""
    print("Grade Group Analysis:")
    bins = [0, 5, 10, 15, 20]
    labels = ['0-4', '5-9', '10-14', '15-20']
    df['grade_group'] = pd.cut(df['G3'], bins=bins, labels=labels, right=False)
    group_stats = df.groupby('grade_group').agg(
        count=('G3', 'size'),
        avg_studytime=('studytime', 'mean'),
        avg_absences=('absences', 'mean'),
        avg_failures=('failures', 'mean'),
        avg_age=('age', 'mean')
    )
    print(group_stats)
    plt.figure(figsize=(10, 6))
    group_stats['count'].plot(kind='bar')
    plt.title('Number of Students by Final Grade Group')
    plt.xlabel('Grade Group (G3)')
    plt.ylabel('Number of Students')
    plt.tight_layout()
    plt.savefig('plots/grade_group_counts.png')
    plt.close()

# ================================
# 10. KEY INSIGHTS AND RECOMMENDATIONS
# ================================
def generate_insights():
    """Generate key insights from the analysis"""
    print("\n10. KEY INSIGHTS AND RECOMMENDATIONS")
    print("-" * 50)
    pass_rate = df['pass'].mean() * 100
    avg_study = df['studytime'].mean()
    avg_abs = df['absences'].mean()
    avg_fail = df['failures'].mean()
    print("KEY FINDINGS:")
    print(f"• Overall pass rate (G3>=10): {pass_rate:.1f}%")
    print(f"• Average study time: {avg_study:.2f}")
    print(f"• Average absences: {avg_abs:.1f}")
    print(f"• Average past failures: {avg_fail:.2f}")
    # Most important features (by correlation)
    correlations = df.select_dtypes(include=[np.number]).corr()['G3'].sort_values(ascending=False)
    print("\nMOST IMPORTANT FEATURES (by correlation with final grade):")
    for i, (feature, corr) in enumerate(list(correlations.items())[1:6]):  # Skip G3 itself
        print(f"{i+1}. {feature}: {corr:.3f}")
    print("\nRECOMMENDATIONS:")
    print("• Monitor and support students with high past failures.")
    print("• Encourage regular study habits and minimize absences.")
    print("• Early intervention for students underperforming in G1/G2.")

# ================================
# ADDITIONAL UTILITY FUNCTIONS
# ================================

def export_summary_report():
    """Export a summary report of the analysis"""
    numeric_df = df.select_dtypes(include=[np.number])  # Only numeric columns
    summary = {
        'Dataset Info': {
            'Total Students': len(df),
            'Total Features': len(df.columns),
            'Pass Rate': f"{df['pass'].mean()*100:.1f}%",
            'Average Study Time': f"{df['studytime'].mean():.2f}",
            'Average Absences': f"{df['absences'].mean():.1f}"
        },
        'Key Statistics': df.describe().to_dict(),
        'Missing Values': df.isnull().sum().to_dict(),
        'Correlation with G3': numeric_df.corr()['G3'].to_dict()
    }
    return summary

# ================================
# MAIN - Redirect ALL PRINT to report.txt
# ================================
if __name__ == "__main__":
    with open("report.txt", "w") as f, redirect_stdout(f):

        print("=" * 60)
        print("STUDENT PERFORMANCE DATASET - COMPREHENSIVE EDA")
        print("Author: Cholpon Zhakshylykova")
        print("=" * 60)

        # 1. DATASET OVERVIEW
        print("\n1. DATASET OVERVIEW")
        print("-" * 30)
        print(f"Dataset Shape: {df.shape}")
        print(f"Number of Students: {df.shape[0]}")
        print(f"Number of Features: {df.shape[1]}")
        print("\nColumn Information:")
        print(df.info())
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nColumn Names:")
        print(df.columns.tolist())

        # 2. FEATURE DESCRIPTIONS
        print("\n2. FEATURE DESCRIPTIONS")
        print("-" * 30)
        feature_descriptions = {
            'school': 'Student’s school (binary: "GP" - Gabriel Pereira or "MS" - Mousinho da Silveira)',
            'sex': 'Student’s sex (binary: "F" - female or "M" - male)',
            'age': 'Student’s age (numeric: from 15 to 22)',
            'address': 'Student’s home address type (binary: "U" - urban or "R" - rural)',
            'famsize': 'Family size (binary: "LE3" - <= 3 or "GT3" - >3)',
            'Pstatus': 'Parent’s cohabitation status (binary: "T" - together or "A" - apart)',
            'Medu': "Mother's education (0 to 4)",
            'Fedu': "Father's education (0 to 4)",
            'Mjob': "Mother's job (nominal)",
            'Fjob': "Father's job (nominal)",
            'reason': "Reason to choose this school (nominal)",
            'guardian': "Student’s guardian (nominal)",
            'traveltime': 'Home to school travel time (1 to 4)',
            'studytime': 'Weekly study time (1 to 4)',
            'failures': 'Number of past class failures (numeric: n if 1<=n<3, else 4)',
            'schoolsup': 'Extra educational support (binary: yes or no)',
            'famsup': 'Family educational support (binary: yes or no)',
            'paid': 'Extra paid classes (binary: yes or no)',
            'activities': 'Extra-curricular activities (binary: yes or no)',
            'nursery': 'Attended nursery school (binary: yes or no)',
            'higher': 'Wants to take higher education (binary: yes or no)',
            'internet': 'Internet access at home (binary: yes or no)',
            'romantic': 'With a romantic relationship (binary: yes or no)',
            'famrel': 'Quality of family relationships (1 to 5)',
            'freetime': 'Free time after school (1 to 5)',
            'goout': 'Going out with friends (1 to 5)',
            'Dalc': 'Workday alcohol consumption (1 to 5)',
            'Walc': 'Weekend alcohol consumption (1 to 5)',
            'health': 'Current health status (1 to 5)',
            'absences': 'Number of school absences',
            'G1': 'First period grade (0 to 20)',
            'G2': 'Second period grade (0 to 20)',
            'G3': 'Final grade (0 to 20)'
        }
        for feature, description in feature_descriptions.items():
            if feature in df.columns:
                print(f"• {feature}: {description}")

        # 3. DATA QUALITY ASSESSMENT
        print("\n3. DATA QUALITY ASSESSMENT")
        print("-" * 30)
        print("Missing Values:")
        missing_values = df.isnull().sum()
        print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values found!")
        print("\nDuplicate Rows:")
        duplicates = df.duplicated().sum()
        print(f"Number of duplicate rows: {duplicates}")
        print("\nData Types:")
        print(df.dtypes)

        # 4. DESCRIPTIVE STATISTICS
        print("\n4. DESCRIPTIVE STATISTICS")
        print("-" * 30)
        print("Numerical Features Summary:")
        print(df.describe())
        print("\nCategorical Features Summary:")
        categorical_cols = ['sex', 'school', 'address', 'famsize', 'Pstatus', 'schoolsup', 'famsup', 'paid', 'activities', 'higher']
        for col in categorical_cols:
            if col in df.columns:
                print(f"\n{col}:")
                print(df[col].value_counts())
                percentages = (df[col].value_counts(normalize=True) * 100).round(2)
                print(f"Percentage distribution (%):\n{percentages}")

        # 5. TARGET VARIABLE ANALYSIS
        print("\n5. TARGET VARIABLE ANALYSIS")
        print("-" * 30)
        if 'G3' in df.columns:
            g3_counts = df['G3'].value_counts().sort_index()
            print(f"Final Grade (G3) Distribution:\n{g3_counts}")
            print(f"Mean final grade: {df['G3'].mean():.2f}")

        # ================================
        # RUN ALL ANALYSES
        # ================================
        print("\n" + "="*60)
        print("RUNNING COMPREHENSIVE ANALYSIS...")
        print("="*60)
        try:
            create_comprehensive_plots()
            perform_statistical_tests()
            advanced_analysis()
            grade_group_analysis()
            generate_insights()
            print("\n" + "="*60)
            print("EDA COMPLETED SUCCESSFULLY!")
            print("="*60)
        except Exception as e:
            print(f"Error in analysis: {e}")
            print("Please ensure the dataset is loaded correctly and all required libraries are installed.")

        # ADDITIONAL REPORT
        print("\nFINAL SUMMARY:")
        summary = export_summary_report()
        for key, value in summary['Dataset Info'].items():
            print(f"• {key}: {value}")
        print("\nKey Statistics:")
        for stat_key, stat_val in summary['Key Statistics'].items():
            print(f"{stat_key}: {stat_val}")
        print("\nMissing Values:")
        for mkey, mval in summary['Missing Values'].items():
            print(f"{mkey}: {mval}")
        print("\nCorrelation with G3:")
        for ckey, cval in summary['Correlation with G3'].items():
            print(f"{ckey}: {cval:.3f}")
        print("\n" + "="*60)
        print("END OF ANALYSIS")
        print("Author: Cholpon Zhakshylykova")
        print("Dataset: Student Performance (UCI)")
        print("="*60)
