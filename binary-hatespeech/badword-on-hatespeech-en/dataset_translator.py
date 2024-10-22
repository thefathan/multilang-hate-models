import pandas as pd
from deep_translator import GoogleTranslator

# Load the CSV file
df = pd.read_csv('/nas.dbms/fathan/test/multilang-hate-models/data_preprocessed.csv')

# Initialize the translator
translator = GoogleTranslator(source='id', target='en')

# Function to translate text
def translate_text(text):
    try:
        return translator.translate(text)
    except Exception as e:
        print(f"Error translating: {text} - {e}")
        return text  # Return original text if translation fails

# Apply the translation to the 'text' column
df['text_translated'] = df['text'].apply(translate_text)

# Save the translated DataFrame to a new CSV
df.to_csv('data_preprocessed_en_translated.csv', index=False)

print("Translation complete. Saved as 'data_preprocessed_en_translated.csv'")
