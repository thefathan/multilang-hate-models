import pandas as pd
from deep_translator import GoogleTranslator

# Load the CSV file
frac_sample = 0.25
df1 = pd.read_csv('/nas.dbms/fathan/test/multilang-hate-models/binary-hatespeech/badword-on-hatespeech-en/negative_hatespeech_with_sexual_words.csv').sample(frac=frac_sample, random_state=42)
df2 = pd.read_csv('/nas.dbms/fathan/test/multilang-hate-models/binary-hatespeech/badword-on-hatespeech-en/negative_hatespeech_without_sexual_words.csv').sample(frac=frac_sample, random_state=42)
df3 = pd.read_csv('/nas.dbms/fathan/test/multilang-hate-models/binary-hatespeech/badword-on-hatespeech-en/positive_hatespeech_with_sexual_words.csv').sample(frac=frac_sample, random_state=42)
df4 = pd.read_csv('/nas.dbms/fathan/test/multilang-hate-models/binary-hatespeech/badword-on-hatespeech-en/positive_hatespeech_without_sexual_words.csv').sample(frac=frac_sample, random_state=42)

# Initialize the translator
translator = GoogleTranslator(source='en', target='ms')

# Function to translate text
def translate_text(text):
    try:
        return translator.translate(text)
    except Exception as e:
        print(f"Error translating: {text} - {e}")
        return text  # Return original text if translation fails

# Apply the translation to the 'text' column
df1['text_translated'] = df1['text'].apply(translate_text)
df2['text_translated'] = df2['text'].apply(translate_text)
df3['text_translated'] = df3['text'].apply(translate_text)
df4['text_translated'] = df4['text'].apply(translate_text)

# Save the translated DataFrame to a new CSV
df1.to_csv('/nas.dbms/fathan/test/multilang-hate-models/binary-hatespeech/badword-on-hatespeech-en/my_translated/negative_hatespeech_with_sexual_words_my_translated_downscalled.csv', index=False)
print("Translation complete. Saved as 'negative_hatespeech_with_sexual_words_my_translated_downscalled.csv'")

df2.to_csv('/nas.dbms/fathan/test/multilang-hate-models/binary-hatespeech/badword-on-hatespeech-en/my_translated/negative_hatespeech_without_sexual_words_my_translated_downscalled.csv', index=False)
print("Translation complete. Saved as 'negative_hatespeech_without_sexual_words_my_translated_downscalled.csv'")

df3.to_csv('/nas.dbms/fathan/test/multilang-hate-models/binary-hatespeech/badword-on-hatespeech-en/my_translated/positive_hatespeech_with_sexual_words_my_translated_downscalled.csv', index=False)
print("Translation complete. Saved as 'positive_hatespeech_with_sexual_words_my_translated_downscalled.csv'")

df4.to_csv('/nas.dbms/fathan/test/multilang-hate-models/binary-hatespeech/badword-on-hatespeech-en/my_translated/positive_hatespeech_without_sexual_words_my_translated_downscalled.csv', index=False)
print("Translation complete. Saved as 'positive_hatespeech_without_sexual_words_my_translated_downscalled.csv'")
