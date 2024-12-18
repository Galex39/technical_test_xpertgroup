import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data Preparation Function
def prepare_data(df):
    print("Iniciando preparación de datos...")
    # Create a deep copy of the DataFrame to avoid SettingWithCopyWarning
    df_cleaned = df.copy(deep=True)

    # Remove duplicate rows
    duplicates_removed = df_cleaned.duplicated().sum()
    print(f"Se encontraron {duplicates_removed} filas duplicadas.")
    df_cleaned = df_cleaned.drop_duplicates().reset_index(drop=True)
    
    # Convert Date column to datetime
    df_cleaned.loc[:, 'Date'] = pd.to_datetime(df_cleaned['Date'])
    print("Columna 'Date' convertida a tipo datetime.")

    # Calculate total sales value
    df_cleaned.loc[:, 'Total_Sales'] = df_cleaned['Units_Sold'] * df_cleaned['Unit_Price']
    print("Columna 'Total_Sales' calculada correctamente.")

    return df_cleaned, duplicates_removed

# Handle Missing Values
def handle_missing_values(df):
    print("Manejando valores nulos...")
    # Resumen inicial de valores nulos
    missing_summary = df.isnull().sum()
    print("Resumen inicial de valores nulos:")
    print(missing_summary)
    
    # Manejo de valores numéricos
    handled_nulls = {}
    for column in df.select_dtypes(include=np.number).columns:
        if df[column].isnull().sum() > 0:
            median_value = df[column].median()
            df[column].fillna(median_value, inplace=True)
            handled_nulls[column] = f"Mediana ({median_value})"
            print(f"Valores nulos en '{column}' rellenados con la mediana ({median_value}).")

    # Manejo de valores categóricos
    for column in df.select_dtypes(include='object').columns:
        if df[column].isnull().sum() > 0:
            mode_value = df[column].mode()[0]
            df[column].fillna(mode_value, inplace=True)
            handled_nulls[column] = f"Moda ({mode_value})"
            print(f"Valores nulos en '{column}' rellenados con la moda ('{mode_value}').")

    return df, missing_summary, handled_nulls

# Descriptive Statistics Function
def generate_descriptive_stats(df):
    print("Generando estadísticas descriptivas...")
    desc_stats = df[['Units_Sold', 'Unit_Price', 'Total_Sales']].describe()
    print("Estadísticas descriptivas generales calculadas.")

    category_stats = df.groupby('Category').agg({
        'Units_Sold': ['mean', 'median', 'std'],
        'Unit_Price': ['mean', 'median', 'std'],
        'Total_Sales': ['mean', 'median', 'std']
    })
    print("Estadísticas descriptivas por categoría calculadas.")

    return desc_stats, category_stats

# Correlation Analysis Function
def analyze_correlations(df):
    print("Analizando correlaciones...")
    correlation_matrix = df[['Units_Sold', 'Unit_Price', 'Total_Sales']].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap of Sales Variables')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()
    print("Heatmap de correlación generado y guardado como 'correlation_heatmap.png'.")

    return correlation_matrix

# Outlier Detection Function
def detect_outliers(df):
    print("Detectando valores atípicos...")

    def find_outliers(column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        return outliers, lower_bound, upper_bound

    units_sold_outliers, units_sold_lower, units_sold_upper = find_outliers('Units_Sold')
    total_sales_outliers, total_sales_lower, total_sales_upper = find_outliers('Total_Sales')

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(x=df['Units_Sold'])
    plt.title('Units Sold Boxplot')

    plt.subplot(1, 2, 2)
    sns.boxplot(x=df['Total_Sales'])
    plt.title('Total Sales Boxplot')

    plt.tight_layout()
    plt.savefig('outliers_boxplot.png')
    plt.close()
    print("Boxplots de valores atípicos generados y guardados como 'outliers_boxplot.png'.")

    return {
        'Units_Sold_Outliers': {
            'outliers': units_sold_outliers,
            'lower_bound': units_sold_lower,
            'upper_bound': units_sold_upper
        },
        'Total_Sales_Outliers': {
            'outliers': total_sales_outliers,
            'lower_bound': total_sales_lower,
            'upper_bound': total_sales_upper
        }
    }

# Category-wise Analysis Function
def category_analysis(df):
    # Sales performance by category
    category_performance = df.groupby('Category').agg({
        'Units_Sold': 'sum',
        'Total_Sales': 'sum'
    }).sort_values('Total_Sales', ascending=False)
    
    # Visualization of category performance
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    category_performance['Units_Sold'].plot(kind='bar')
    plt.title('Total Units Sold by Category')
    plt.xlabel('Category')
    plt.ylabel('Total Units')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    category_performance['Total_Sales'].plot(kind='bar')
    plt.title('Total Sales by Category')
    plt.xlabel('Category')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('category_performance.png')
    plt.close()
    
    return category_performance

# Main Execution
def main():
    print("Iniciando ejecución principal...")
    # Read the CSV file
    print("Leyendo archivo CSV...")
    df = pd.read_csv('sales_data.csv')

    # Prepare the data
    df_cleaned, duplicates_removed = prepare_data(df)

    # Handle missing values
    df_cleaned, missing_summary, handled_nulls = handle_missing_values(df_cleaned)

    # Descriptive statistics
    # Run analyses
    desc_stats, category_stats = generate_descriptive_stats(df_cleaned)
    correlation_matrix = analyze_correlations(df_cleaned)
    outliers = detect_outliers(df_cleaned)
    category_performance = category_analysis(df_cleaned)
    
    # Print results
    print("============================================= Report =============================================")
    print("Resumen de Valores Nulos:")
    print(missing_summary.to_string())
    print("Manejo de Valores Nulos:")
    for col, method in handled_nulls.items():
        print(f"   - Columna '{col}': {method}")
    print("Duplicados Eliminados:")
    print(f"   - Total eliminados: {duplicates_removed}")
    print("Overall Descriptive Statistics:")
    print(desc_stats)
    print("\nCategory-wise Statistics:")
    print(category_stats)
    print("\nCorrelation Matrix:")
    print(correlation_matrix)
    print("\nOutlier Analysis Summary:")
    print(f"Units Sold Outliers Range: [{outliers['Units_Sold_Outliers']['lower_bound']}, {outliers['Units_Sold_Outliers']['upper_bound']}]")
    print(f"Total Sales Outliers Range: [{outliers['Total_Sales_Outliers']['lower_bound']}, {outliers['Total_Sales_Outliers']['upper_bound']}]")
    print("\nCategory Performance:")
    print(category_performance)

# Run the main function
if __name__ == "__main__":
    main()
