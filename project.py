# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import re
from collections import Counter
from nltk.corpus import stopwords

# Download the NLTK stopwords dataset
import nltk
nltk.download('stopwords')

# Load dataset
df = pd.read_csv('best_buy_laptops_2024.csv')

# Check for missing values
missing_values = df.isnull().sum()

# Calculate total number of missing cells
total_missing_cells = missing_values.sum()

# Create a DataFrame to store the results
missing_values_df = pd.DataFrame(missing_values, columns=['Missing Values'])

# Display the DataFrame
print("Number of Missing Values in Each Column:")
print(missing_values_df)

# Display total number of missing cells
print("\nTotal Number of Missing Cells:", total_missing_cells)

# Create a bar graph for missing value
plt.figure(figsize=(10, 6))
missing_values.plot(kind='bar', color='skyblue')
plt.title('Number of Missing Values in Each Column')
plt.xlabel('Columns')
plt.ylabel('Number of Missing Values')
plt.xticks(rotation=45, ha='right')
plt.text(len(missing_values)-1, max(missing_values), f'Total missing: {total_missing_cells}', ha='right', va='top', fontsize=10, color='red')
plt.savefig('missine_values.png')  # Save bar plot as image
plt.close() 


# Fill missing values with the mean of numerical columns rounded to two decimal places
numeric_columns = df.select_dtypes(include=['number']).columns
mean_rounded = df[numeric_columns].mean().round(2)
df[numeric_columns] = df[numeric_columns].fillna(mean_rounded)

# Fill missing categorical values with a specific value
categorical_columns = df.select_dtypes(exclude=['number']).columns
df[categorical_columns] = df[categorical_columns].fillna("Unknown")

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Convert price column to numeric format
df['offers/price'] = pd.to_numeric(df['offers/price'], errors='coerce')

# Standardize or normalize numerical features (e.g., depth and width)
scaler = StandardScaler()
df[['depth', 'width']] = scaler.fit_transform(df[['depth', 'width']])

# Data validation and sanity checks
df = df[df['offers/price'] > 0]

# Summary statistics
summary_stats = df.describe()

# Create a new DataFrame with selected columns
specs_df = df[['brand', 'model', 'features/0/description', 'features/1/description']].copy()
# Rename column features/0/description to featuresMain
specs_df.rename(columns={'features/0/description': 'featuresMain'}, inplace=True)
# Rename column features/1/description to featSecondary
specs_df.rename(columns={'features/1/description': 'featSecondary'}, inplace=True)

# Get the data types of columns in the DataFrame
column_types = specs_df.dtypes
# Print data types of each column
for column in specs_df.columns:
    print( f"{column}: {specs_df[column].dtype}" )
# Display information about the DataFrame
print( specs_df.info() )

# Convert column featuresMain and featSecondary to strings
specs_df['featuresMain'] = specs_df['featuresMain'].astype("string")
specs_df['featSecondary'] = specs_df['featSecondary'].astype("string")
# Print data types of each column
for column in specs_df.columns:
    print( f"{column}: {specs_df[column].dtype}" )
    
    

# Define a function to extract processor-related information from a text
def extract_processors(text):
    # Define keywords related to processors
    processor_keywords = ['processor', 'CPU', 'core']
    
    # Compile regex pattern to match processor-related keywords
    pattern = re.compile(r'\b(?:' + '|'.join(processor_keywords) + r')\b', flags=re.IGNORECASE)
    
    # Find all matches in the text
    matches = re.findall(pattern, text)
    
    # Return unique matches
    return list(set(matches))

# Apply the function to each description in the 'featuresMain' column
processor_mentions = specs_df['featuresMain'].apply(extract_processors)

# Display the processor mentions for each description
for i, processors in enumerate(processor_mentions):
    print(f"Description {i+1}: {processors}")

    # Define a function to check if a text contains mentions of processors
def contains_processors(text):
    # Define keywords related to processors
    processor_keywords = ['processor', 'CPU', 'core']
    
    # Compile regex pattern to match processor-related keywords
    pattern = re.compile(r'\b(?:' + '|'.join(processor_keywords) + r')\b', flags=re.IGNORECASE)
    
    # Check if the text contains any matches
    if re.search(pattern, text):
        return True
    else:
        return False

# Apply the function to each description in the 'featuresMain' column
rows_with_processors = specs_df['featuresMain'].apply(contains_processors)

# Count the number of rows containing mentions of processors
total_rows_with_processors = rows_with_processors.sum()

# Print the total number of rows containing mentions of processors
print(f"Total number of rows containing mentions of processors: {total_rows_with_processors}")

# Define a function to check if a text contains mentions of processors
def contains_processors(text):
    # Define keywords related to processors
    processor_keywords = ['processor', 'CPU', 'core']
    
    # Compile regex pattern to match processor-related keywords
    pattern = re.compile(r'\b(?:' + '|'.join(processor_keywords) + r')\b', flags=re.IGNORECASE)
    
    # Check if the text contains any matches
    if re.search(pattern, text):
        return True
    else:
        return False

# Apply the function to each description in the 'featuresMain' column
rows_with_processors = specs_df['featSecondary'].apply(contains_processors)

# Count the number of rows containing mentions of processors
total_rows_with_processors = rows_with_processors.sum()

# Print the total number of rows containing mentions of processors
print(f"Total number of rows containing mentions of processors: {total_rows_with_processors}")


# Define a function to extract RAM-related information from a text
def extract_ram(text):
    # Define keywords related to RAM
    ram_keywords = ['RAM', 'memory', 'DDR'] 
    # Compile regex pattern to match RAM-related keywords
    pattern = re.compile(r'\b(?:' + '|'.join(ram_keywords) + r')\b', flags=re.IGNORECASE)   
    # Check if the text contains any matches
    if re.search(pattern, text):
        return True
    else:
        return False

# Apply the function to each description in the 'featuresMain' column
rows_with_RAM = specs_df['featuresMain'].apply(extract_ram)
total_rows_with_RAM = rows_with_RAM.sum()
print(f"Total number of rows containing mentions of RAM: {total_rows_with_RAM}")


# Get the English stopwords
stop_words = set(stopwords.words('english'))
# Combine all descriptions from both columns into a single string
all_descriptions = ' '.join( specs_df['featuresMain'] + ' ' + specs_df['featSecondary'] )

# Define a function to tokenize the combined string into words, considering specific phrases, numbers, and patterns
def tokenize_text(text):
    # Define regular expressions to match specific Windows versions and patterns
    windows_versions = ['Windows 11', 'Windows 10', 'Windows 8', 'Windows 7', 'Windows Vista', 'Windows XP']
    resolution_pattern = re.compile(r'\b\d+\s*x\s*\d+\b')
    side_by_side_pattern = re.compile(r'\b(side\s*by\s*side)\b', flags=re.IGNORECASE)

    # Compile regular expressions to match Windows versions
    version_patterns = [re.compile(re.escape(version), flags=re.IGNORECASE) for version in windows_versions]

    # Replace specific Windows versions with a placeholder token
    for pattern, version in zip(version_patterns, windows_versions):
        text = re.sub(pattern, f'WINDOWS_VERSION_{version.split()[-1]}', text)
    # Replace specific resolution patterns with a placeholder token
    text = re.sub(resolution_pattern, 'RESOLUTION_PATTERN', text)
    # Replace occurrences of 'side by side' with a single token
    text = re.sub(side_by_side_pattern, 'SIDE_BY_SIDE_PATTERN', text)

    # Tokenize the modified text into words
    words = re.findall(r'\b\w+\b', text.lower())
    # Exclude specific words 'like' and 'need'
    words = [word for word in words if word not in stop_words and word not in ['like', 'need', 'comes', '16']]
    # Replace occurrences of 'resolution' and 'resolution_pattern' with a single token
    words = ['RESOLUTION' if word in ['resolution', 'resolution_pattern'] else word for word in words]

    return words

# Tokenize the combined string into words, considering specific phrases and numbers
words = tokenize_text(all_descriptions)
# Filter out stopwords
filtered_words = [word for word in words if word not in stop_words]

# Count the frequency of each word
word_counts = Counter(filtered_words)
# Get the top 20 most common words
top_20_words = word_counts.most_common(20)

# Print the top 20 most common words
print("Top 20 most used words in featuresMain and featSecondary (excluding stopwords):")
for word, count in top_20_words:
    print(f"{word}: {count}")

    # Extract words and counts for plotting
words = [word[0] for word in top_20_words]
counts = [word[1] for word in top_20_words]

# Plot the bar plot
plt.figure(figsize=(8, 4))
plt.bar(words, counts, color='skyblue')
plt.title('Top 20 most used words to sell')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.savefig('top_20_used_words_to_sell.png')  # Save bar plot as image
plt.close() 


# Histogram of price
plt.figure(figsize=(8, 6))
sns.histplot(df['offers/price'], bins=20, kde=True)
plt.title('Distribution of Price')
plt.xlabel('Price (USD)')
plt.ylabel('Frequency')
plt.savefig('price_distribution.png')  # Save histogram as image
plt.close()

# Calculate average price per brand
average_price_per_brand = df.groupby('brand')['offers/price'].mean().sort_values(ascending=False)

# Plotting the bar graph
plt.figure(figsize=(10, 6))
average_price_per_brand.plot(kind='bar', color='skyblue')
plt.title('Average Price per Brand after cleaning')
plt.xlabel('Brand')
plt.ylabel('Average Price')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.savefig('average_price_per_brand.png')  # Save bar graph as image
plt.close()

# Calculate satisfaction to price ratio
df['satisfaction_to_price_ratio'] = df['aggregateRating/ratingValue'] / df['offers/price']

# Plotting the bar graph
plt.figure(figsize=(10, 6))
df.groupby('brand')['satisfaction_to_price_ratio'].mean().sort_values(ascending=False).plot(kind='bar', color='skyblue')
plt.title('Satisfaction to Price Ratio per Brand after cleaning')
plt.xlabel('Brand')
plt.ylabel('Satisfaction to Price Ratio')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.savefig('satisfaction_to_price_ratio.png')  # Save bar graph as image
plt.close()


# Scatter plot of price vs. rating
plt.figure(figsize=(8, 6))
sns.scatterplot(x='aggregateRating/ratingValue', y='offers/price', data=df)
plt.title('Scatter Plot of Rating vs. Price')
plt.xlabel('Rating')
plt.ylabel('Price (USD)')
plt.savefig('rating_vs_price_scatter.png')  # Save scatter plot as image
plt.close()

# Calculate satisfaction to price ratio
df['satisfaction_to_price_ratio'] = df['aggregateRating/ratingValue'] / df['offers/price']

# Plotting the bar graph
plt.figure(figsize=(10, 6))
df.groupby('brand')['satisfaction_to_price_ratio'].mean().sort_values(ascending=False).plot(kind='bar', color='skyblue')
plt.title('Satisfaction to Price Ratio per Brand ')
plt.xlabel('Brand')
plt.ylabel('Satisfaction to Price Ratio')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.savefig('satisfaction_to_price_ratio_per_brands.png')  # Savebar graph as image
plt.close()

# Scatter plot of price vs. rating
plt.figure(figsize=(7, 4))
sns.scatterplot(x='aggregateRating/ratingValue', y='offers/price', data=df)
plt.title('Scatter Plot of Rating vs. Price')
plt.xlabel('Rating')
plt.ylabel('Price (USD)')

# Add a red line of best fit
sns.regplot(x='aggregateRating/ratingValue', y='offers/price', data=df, scatter=False, color='red')
plt.savefig('scatter_plot_of_rating_vs_price_red_line.png')  # Savebar graph as image
plt.close()

# coorelation matrix
numeric_df = df.select_dtypes(include=['number'])# Exclude non-numeric columns from correlation calculation
correlation_matrix = numeric_df.corr()# Calculate correlation matrix
#print(correlation_matrix)


# Correlation heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')  # Save heatmap as image
plt.close()

# Count plot of brands
plt.figure(figsize=(10, 6))
sns.countplot(x='brand', data=df)
plt.title('Count of Laptops by Brand')
plt.xlabel('Brand')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.savefig('brand_count_plot.png')  # Save count plot as image
plt.close() 

# Group by brand and calculate aggregate statistics
brand_comparison = df.groupby('brand').agg({
    'aggregateRating/ratingValue': 'mean',  # Average rating per brand
    'aggregateRating/reviewCount': 'sum'    # Total review count per brand
}).reset_index()

# Sort by average rating value in descending order
brand_comparison = brand_comparison.sort_values(by='aggregateRating/ratingValue', ascending=False)

# Visualize the comparison
plt.figure(figsize=(10, 6))
sns.barplot(y='aggregateRating/ratingValue', x='brand', data=brand_comparison, palette='viridis')
plt.title('Average Ratings by Brand')
plt.ylabel('Average Rating')
plt.xlabel('Brand')
plt.xticks(rotation=45)
plt.savefig('average_rating_per_brand.png')  # Save bar plot as image
plt.close() 

# total review count per brand
plt.figure(figsize=(10, 6))
sns.barplot(y='aggregateRating/reviewCount', x='brand', data=brand_comparison, palette='viridis')
plt.title('Total Review Counts by Brand')
plt.ylabel('Total Review Count')
plt.xlabel('Brand')
plt.savefig('total_review_count_per_brand.png')  # Save bar plot as image
plt.close() 

#Predictive modelling
# Data Preprocessing and model evaluation
# Identify features and target variable
features = ['offers/price', 'depth', 'width', 'aggregateRating/reviewCount']
target = 'aggregateRating/ratingValue'

# Separate features and target variable
X = df[features]
y = df[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps for numerical and categorical features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, features)
    ])

# Define models to try
models = [
    ('Linear Regression', LinearRegression()),
    ('Decision Tree', DecisionTreeRegressor(random_state=42)),
    ('Random Forest', RandomForestRegressor(random_state=42))
]

# Evaluate each model using cross-validation
for name, model in models:
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('model', model)])
    cv_scores = cross_val_score(model_pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    print(f"{name}: Mean MSE: {-cv_scores.mean()} +/- {cv_scores.std()}")
    
# Fit and evaluate the best model on the test set
best_model = RandomForestRegressor(random_state=42)
best_model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('model', best_model)])
best_model_pipeline.fit(X_train, y_train)
y_pred = best_model_pipeline.predict(X_test)

print("Test Set Evaluation:")
print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
print(f"R-squared: {r2_score(y_test, y_pred)}")


# feature importance analysis
# Check if the best model supports feature importances
if 'feature_importances_' in dir(best_model_pipeline.named_steps['model']):
    # Get feature importances if available
    feature_importances = best_model_pipeline.named_steps['model'].feature_importances_
    feature_names = X_train.columns
    # Sort feature importances
    sorted_indices = feature_importances.argsort()[::-1]
    sorted_feature_names = feature_names[sorted_indices]
    sorted_feature_importances = feature_importances[sorted_indices]
    # Plot feature importances
    plt.figure(figsize=(15, 5))
    plt.bar(range(len(feature_importances)), sorted_feature_importances)
    plt.xticks(range(len(feature_importances)), sorted_feature_names, rotation=30)
    plt.title('Feature Importances')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.savefig('Feature_importance.png')  # Save bar plot as image
    plt.close() 
else:
    print("Feature importances are not available for this model.")


#Sentiment analysis

# sentiment analysis using the rating and review counts to gauge consumer satisfaction across different brands or models

# Assuming df is your DataFrame containing the dataset
# Calculate sentiment score
df['sentiment_score'] = df['aggregateRating/ratingValue'] * df['aggregateRating/reviewCount']

# Aggregate sentiment scores by brand
brand_sentiment = df.groupby('brand')['sentiment_score'].mean().reset_index()

# Aggregate sentiment scores by model
model_sentiment = df.groupby('model')['sentiment_score'].mean().reset_index()

plt.figure(figsize=(10, 6))
plt.bar(brand_sentiment['brand'], brand_sentiment['sentiment_score'])
plt.xlabel('Brand')
plt.ylabel('Sentiment Score')
plt.title('Average Sentiment Score by Brand')
plt.xticks(rotation=45)
plt.savefig('brand_sentiment.png')  # Save bar plot as image
plt.close() 

# Visualization - Bar chart for model sentiment
plt.figure(figsize=(100, 6))
plt.bar(model_sentiment['model'], model_sentiment['sentiment_score'])
plt.xlabel('Model')
plt.ylabel('Sentiment Score')
plt.title('Average Sentiment Score by Model')
plt.xticks(rotation=90)
plt.savefig('model_sentiment.png')  # Save bar plot as image
plt.close() 

