import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

train_df = pd.read_json('subtaskA/data/subtaskA_train_monolingual.jsonl', lines=True)  
test_df = pd.read_json('subtaskA/data/subtaskA_dev_monolingual.jsonl', lines=True)    

X_train = train_df['text']
y_train = train_df['label']

# Convert text data to numerical features using CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

# Initialize and train a logistic regression model
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

# Make predictions on the test set
X_test = test_df['text']
X_test_vectorized = vectorizer.transform(X_test)
predictions = model.predict(X_test_vectorized)

# Output predictions in the specified format
output_file = 'subtaskA/predictions/predictions.jsonl'  
with open(output_file, 'w', encoding='utf-8') as f:
    for idx in range(len(test_df)):
        text = test_df.loc[idx, 'text']
        pred_label = int(model.predict(vectorizer.transform([text]))[0])
        output_line = f'{{"id": {int(test_df.loc[idx, "id"])}, "label": {pred_label}}}\n'
        f.write(output_line)

print(f'Predictions saved to {output_file}')