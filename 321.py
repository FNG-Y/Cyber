import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

# Set matplotlib style
plt.style.use('default')

# -----------------------------------------------------------------------------
# Task 1: Data Analysis and Visualisation
# -----------------------------------------------------------------------------

print("=" * 60)
print("Task 1: Data Analysis and Visualisation")
print("=" * 60)

# Task 1a: Data Loading and Cleaning
print("\n1a. Data Loading and Cleaning")
print("-" * 40)

# Load data
df = pd.read_csv('./transport_delays.csv')

# Print first 5 rows and shape
print("First 5 rows:")
print(df.head())
print(f"\nDataFrame shape: {df.shape}")

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])
print(f"\nDate column type after conversion: {df['Date'].dtype}")

# Handle missing data
print(f"\nMissing values BEFORE filling:")
print(df.isnull().sum())

# Fill missing values
df['Delay_Type'].fillna('UNKNOWN', inplace=True)
df['Minutes_Delayed'].fillna(0, inplace=True)

print(f"\nMissing values AFTER filling:")
print(df.isnull().sum())

# Ensure Delay_Type is uppercase and stripped of spaces
df['Delay_Type'] = df['Delay_Type'].str.strip().str.upper()
print(f"\nUnique Delay_Types:")
print(df['Delay_Type'].unique())

# Task 1b: Data Visualisation
print("\n1b. Data Visualisation")
print("-" * 40)

# Total Delay Minutes per Type
delay_totals = df.groupby('Delay_Type')['Minutes_Delayed'].sum().sort_values(ascending=False)
print("Total minutes delayed per type:")
for delay_type, total in delay_totals.items():
    print(f"{delay_type}: {total:.0f} minutes")

# Create bar chart
plt.figure(figsize=(12, 6))
bars = plt.bar(delay_totals.index, delay_totals.values, color='steelblue')
plt.title('Total Delay Minutes per Type', fontsize=14, fontweight='bold')
plt.xlabel('Delay Type', fontsize=12)
plt.ylabel('Total Minutes Delayed', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Add value labels to bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height,
             f'{height:.0f}', ha='center', va='bottom')

plt.savefig('delay_minutes_per_type.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nBar chart saved as 'delay_minutes_per_type.png'")

# Create line chart of total Minutes_Delayed per month for 2025
df_2025 = df[df['Date'].dt.year == 2025].copy()
df_2025['Month'] = df_2025['Date'].dt.to_period('M')
monthly_totals = df_2025.groupby('Month')['Minutes_Delayed'].sum()

print("\nTotal minutes delayed per month in 2025:")
for month, total in monthly_totals.items():
    print(f"{month}: {total:.0f} minutes")

# Create line chart
plt.figure(figsize=(14, 6))
plt.plot(range(len(monthly_totals)), monthly_totals.values,
         marker='o', linewidth=2, markersize=6, color='darkred')
plt.title('Total Delay Minutes per Month (2025)', fontsize=14, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Total Minutes Delayed', fontsize=12)
plt.xticks(range(len(monthly_totals)),
           [str(month).split('-')[1] for month in monthly_totals.index],
           rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Add value labels
for i, value in enumerate(monthly_totals.values):
    plt.text(i, value, f'{value:.0f}', ha='center', va='bottom')

plt.savefig('monthly_delays_2025.png', dpi=300, bbox_inches='tight')
plt.close()
print("Line chart saved as 'monthly_delays_2025.png'")

# -----------------------------------------------------------------------------
# Task 2: Database Management and File Handling
# -----------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Task 2: Database Management and File Handling")
print("=" * 60)

# Task 2a: Database Creation and Data Insertion
print("\n2a. Database Creation and Data Insertion")
print("-" * 40)

# Create SQLite database
conn = sqlite3.connect('TransitDB.db')
cursor = conn.cursor()

# Create table with UNIQUE constraint
cursor.execute('''
CREATE TABLE IF NOT EXISTS Delays (
    Date DATE,
    Corridor TEXT,
    Delay_Type TEXT,
    Minutes_Delayed INTEGER,
    UNIQUE(Date, Corridor, Delay_Type)
)
''')

# Insert data, respecting UNIQUE constraint
rows_inserted = 0
total_rows = len(df)

for _, row in df.iterrows():
    try:
        cursor.execute('''
        INSERT INTO Delays (Date, Corridor, Delay_Type, Minutes_Delayed)
        VALUES (?, ?, ?, ?)
        ''', (row['Date'].strftime('%Y-%m-%d'), row['Corridor'],
              row['Delay_Type'], int(row['Minutes_Delayed'])))
        rows_inserted += 1
    except sqlite3.IntegrityError:
        # Skip duplicates
        pass

conn.commit()
print(f"Total rows processed: {total_rows}")
print(f"Rows inserted into database: {rows_inserted}")

# Task 2b: Querying the Database
print("\n2b. Querying the Database")
print("-" * 40)

# Total Delay Minutes in 2025
cursor.execute('''
SELECT SUM(Minutes_Delayed) FROM Delays
WHERE Date LIKE '2025-%'
''')
total_2025 = cursor.fetchone()[0]
print(f"Total minutes delayed in 2025: {total_2025:.0f}")

# Top five corridors in 2025
cursor.execute('''
SELECT Corridor, SUM(Minutes_Delayed) as Total_Delays
FROM Delays
WHERE Date LIKE '2025-%'
GROUP BY Corridor
ORDER BY Total_Delays DESC
LIMIT 5
''')
top_corridors = cursor.fetchall()

print("\nTop five corridors with highest delays in 2025:")
for i, (corridor, total) in enumerate(top_corridors, 1):
    print(f"{i}. {corridor}: {total:.0f} minutes")

# Task 2c: Exporting Results
print("\n2c. Exporting Results")
print("-" * 40)

# Create DataFrame from top corridors
top5_df = pd.DataFrame(top_corridors, columns=['Corridor', 'Total_Delays'])
print("Top 5 corridors DataFrame:")
print(top5_df)

# Save to CSV
top5_df.to_csv('top5_corridors.csv', index=False)
print("\nTop 5 corridors saved to 'top5_corridors.csv'")

# Close database connection
conn.close()

# -----------------------------------------------------------------------------
# Task 3: Neural Network (PyTorch)
# -----------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Task 3: Neural Network (PyTorch)")
print("=" * 60)

# Task 3a: Data & Reproducibility
print("\n3a. Data & Reproducibility")
print("-" * 40)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate synthetic features X (200 samples, 4 features)
n_samples = 200
n_features = 4

# Generate features with realistic distributions
np.random.seed(42)
X = np.zeros((n_samples, n_features))
X[:, 0] = np.random.normal(45, 10, n_samples)  # Average speed (km/h)
X[:, 1] = np.random.poisson(3, n_samples)  # Incidents reported
X[:, 2] = np.random.exponential(5, n_samples)  # Rainfall (mm)
X[:, 3] = np.random.uniform(0, 10, n_samples)  # Event intensity

# Generate binary target labels y
# High delay day (1) is more likely with: lower speed, more incidents, more rainfall, higher event intensity
logits = (-0.05 * X[:, 0] + 0.8 * X[:, 1] + 0.3 * X[:, 2] + 0.5 * X[:, 3] - 5)
probabilities = 1 / (1 + np.exp(-logits))
y = np.random.binomial(1, probabilities)

print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

# Task 3b: Model Definition and Initialization
print("\n3b. Model Definition and Initialization")
print("-" * 40)


class DelayRiskNN(nn.Module):
    def __init__(self):
        super(DelayRiskNN, self).__init__()
        self.hidden = nn.Linear(4, 8)  # Input: 4 features, Hidden: 8 neurons
        self.output = nn.Linear(8, 1)  # Output: 1 neuron

    def forward(self, x):
        x = torch.relu(self.hidden(x))  # ReLU activation
        x = torch.sigmoid(self.output(x))  # Sigmoid activation for binary classification
        return x


# Instantiate model, loss function, and optimizer
model = DelayRiskNN()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.05)

print("Model architecture:")
print(model)

# Task 3c: Training the Model
print("\n3c. Training the Model")
print("-" * 40)

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Training loop
n_epochs = 10
loss_history = []

print("Training loss per epoch:")
for epoch in range(n_epochs):
    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Store loss
    loss_history.append(loss.item())

    # Print progress
    print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}")

# Task 3d: Loss Visualisation
print("\n3d. Loss Visualisation")
print("-" * 40)

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_epochs + 1), loss_history, marker='o', linewidth=2, markersize=6, color='purple')
plt.title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (BCELoss)', fontsize=12)
plt.xticks(range(1, n_epochs + 1))
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Add value labels
for i, loss_value in enumerate(loss_history):
    plt.text(i + 1, loss_value, f'{loss_value:.4f}', ha='center', va='bottom')

plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
plt.close()
print("Loss visualization saved as 'training_loss.png'")

print("\n" + "=" * 60)
print("All tasks completed successfully!")
print("=" * 60)