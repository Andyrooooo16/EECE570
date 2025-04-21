import pandas as pd
import plotly.express as px
import plotly.io as pio

# Set Plotly to open charts in the browser
pio.renderers.default = "browser"

# Define file paths
train_csv_path = "../data/Training_Set/RFMiD_Training_Labels.csv"
eval_csv_path = "../data/Evaluation_Set/RFMiD_Validation_Labels.csv"
test_csv_path = "../data/Test_Set/RFMiD_Testing_Labels.csv"

# Load datasets
df_train = pd.read_csv(train_csv_path)
df_eval = pd.read_csv(eval_csv_path)
df_test = pd.read_csv(test_csv_path)

# Define columns to drop
non_disease_cols = ['ID', 'Disease_Risk']

# Function to generate disease count plot
def plot_disease_distribution(df, title):
    disease_df = df.drop(columns=non_disease_cols)
    disease_counts = disease_df.sum().sort_values(ascending=False)

    print(f"Top 5 most common diseases in {title.lower()}:")
    print(disease_counts.head(5))

    fig = px.bar(
        disease_counts.head(10),
        title=f"Top 10 Most Common Retinal Diseases in {title}",
        labels={'value': 'Number of Samples', 'index': 'Disease'},
        template='plotly_white'
    )
    fig.update_layout(xaxis_tickangle=-45)
    fig.show()

# Plot training data
plot_disease_distribution(df_train, "Training Set")

# Plot evaluation data
plot_disease_distribution(df_eval, "Evaluation Set")

# Plot test data
plot_disease_distribution(df_test, "Test Set")

