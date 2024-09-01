import streamlit as st
import pandas as pd
import json
import folium
from folium.plugins import HeatMapWithTime
from streamlit_folium import st_folium
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import nltk
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import spacy
import string
from nltk.corpus import wordnet as wn
import ast

@st.cache_data
def download_nltk_resources():
    nltk_resources = [
        'vader_lexicon', 'punkt', 'punkt_tab', 'stopwords', 
        'averaged_perceptron_tagger', 'maxent_ne_chunker', 
        'words', 'wordnet', 'omw-1.4'
    ]
    for resource in nltk_resources:
        try:
            nltk.data.find(f'corpora/{resource}')
        except LookupError:
            nltk.download(resource)

download_nltk_resources()

# Load SpaCy's small English model
nlp = spacy.load('en_core_web_sm')

# Load Pre-trained Model and Vectorizer
@st.cache_data
def load_model_and_vectorizer():
    classifier = joblib.load('random_forest_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return classifier, vectorizer

classifier, vectorizer = load_model_and_vectorizer()

# App title
st.title("Tweet Analysis Web App")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file is not None:
    try:
        # Step 1: Read the CSV file
        df = pd.read_csv(uploaded_file)

        # Step 2: Parse JSON data from the 'Tweet JSON' column
        df['Tweet JSON'] = df['Tweet JSON'].apply(json.loads)

        # Step 3: Normalize the JSON data
        json_df = pd.json_normalize(df['Tweet JSON'])

        # Step 4: Combine the JSON DataFrame with the original DataFrame if needed
        df = df.drop(columns=['Tweet JSON'])
        final_df = pd.concat([df, json_df], axis=1)
        df = final_df

        st.write("Data Loaded and Normalized:")
        st.write(final_df.head())
        st.write(final_df.columns)

        # Geospatial Analysis with Time-Based Heatmap
        st.write("")
        st.header("Geospatial Analysis with Time-Based Heatmap")

        # Extract latitude and longitude
        final_df['latitude'] = final_df['geo.coordinates'].apply(lambda x: x[0])
        final_df['longitude'] = final_df['geo.coordinates'].apply(lambda x: x[1])

        map_center = [final_df['latitude'].mean(), final_df['longitude'].mean()]
        m = folium.Map(location=map_center, zoom_start=6)

        # Add markers to the map for each tweet in the DataFrame
        for lat, lon in zip(final_df['latitude'], final_df['longitude']):
            if pd.notna(lat) and pd.notna(lon):  # Ensure latitude and longitude are valid numbers
                folium.Marker(location=[lat, lon]).add_to(m)

        # Streamlit title
        st.title("Tweet Locations Map")

        # Display the map in the Streamlit app
        st_folium(m)

        # Sentiment Analysis
        st.write("")
        st.header("Sentiment Analysis")

        def clean_tweet(text):
            text = re.sub(r'http\S+', '', text)  # Remove URLs
            text = re.sub(r'@\w+', '', text)  # Remove mentions
            text = re.sub(r'#\w+', '', text)  # Remove hashtags
            text = re.sub(r'[^A-Za-z0-9\s]', '', text)  # Remove special characters
            text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
            return text

        df['cleaned_tweet'] = df['full_text'].apply(clean_tweet)
        sid = SentimentIntensityAnalyzer()

        def get_sentiment_score(text):
            sentiment_dict = sid.polarity_scores(text)
            return sentiment_dict

        df['sentiment'] = df['cleaned_tweet'].apply(get_sentiment_score)
        df['sentiment_category'] = df['sentiment'].apply(lambda sentiment: 'Positive' if sentiment['compound'] >= 0.05 else 'Neutral' if sentiment['compound'] > -0.05 else 'Negative')
        sentiment_counts = df['sentiment_category'].value_counts()

        st.write("Sentiment Counts:")
        st.bar_chart(sentiment_counts)

        # Filter Out Negative/Aggressive/Spammy Tweets
        df['is_negative_or_spammy'] = df['sentiment_category'] == 'Negative'
        cleaned_df = df[~df['is_negative_or_spammy']]

        # Prediction with RandomForest model for Relevance/Irrelevance Classification
        st.write("")
        st.header("Classification Prediction for Relevant/Irrelevant Tweets")
        new_data = cleaned_df.copy()
        new_data['cleaned_text'] = new_data['full_text'].apply(clean_tweet)
        new_data_tfidf = vectorizer.transform(new_data['cleaned_text'])
        new_data['predicted_label'] = classifier.predict(new_data_tfidf)
        new_data.to_csv('new_predicted_tweets.csv', index=False)
        st.write("Prediction Completed:")
        st.write(new_data[['full_text', 'predicted_label']].tail())

        # Updated NER for Sightings and Locations using SpaCy
        st.write("")
        st.header("Named Entity Recognition (NER) for Sightings and Locations")
        relevant_tweets_df = new_data[new_data['predicted_label'] == 1].copy()

        # Function to expand keywords with synonyms using WordNet
        def expand_keywords_with_synonyms(keywords):
            synonyms = set(keywords)
            for keyword in keywords:
                for syn in wn.synsets(keyword):
                    for lemma in syn.lemmas():
                        synonyms.add(lemma.name().replace('_', ' '))  # Replace underscores in multi-word synonyms
            return list(synonyms)

        # Initial sighting-specific keywords
        sighting_keywords = ['saw', 'spotted', 'last seen', 'just saw', 'seen near', 'sighted in', 'witnessed', 'noticed']

        # Expand the keywords to include synonyms
        expanded_sighting_keywords = expand_keywords_with_synonyms(sighting_keywords)

        # Create a regex pattern for the expanded keywords
        sighting_keywords_pattern = re.compile(r'\b(' + '|'.join(expanded_sighting_keywords) + r')\b', re.IGNORECASE)

        # Function to split camel case (e.g., #JohnDoeMissing -> John Doe Missing)
        def split_camel_case(hashtag):
            return re.sub(r'([a-z])([A-Z])', r'\1 \2', hashtag)

        # Updated function to identify sightings and handle hashtags
        def identify_sightings_and_locations(text):
            # Find all hashtags
            hashtags = re.findall(r'#\w+', text)
            
            # Split hashtags and add them to the text
            split_hashtags = ' '.join([split_camel_case(h) for h in hashtags])
            
            # Combine original text with split hashtags
            combined_text = text + ' ' + split_hashtags

            # Process combined text with SpaCy NER
            doc = nlp(combined_text)
            
            # Extract locations from text
            locations = [ent.text for ent in doc.ents if ent.label_ == 'GPE']  # 'GPE' stands for Geopolitical Entity
            
            # Check if the text contains expanded sighting keywords and has any identified locations
            if sighting_keywords_pattern.search(combined_text) and locations:
                return locations  # Return the list of locations if keywords are found and locations are present
            else:
                return []  # Return an empty list if no relevant keywords or locations

        # Apply the function to identify locations only for sighting-related tweets
        relevant_tweets_df['sighting_locations'] = relevant_tweets_df['full_text'].apply(identify_sightings_and_locations)

        # Filter out only the rows where sightings were identified (non-empty sighting_locations)
        sightings_df = relevant_tweets_df[relevant_tweets_df['sighting_locations'].apply(lambda x: len(x) > 0)].copy()

        # Combine the sighting locations with the date and time
        sightings_df['location_time'] = sightings_df.apply(lambda row: list(zip(row['sighting_locations'], [row['created_at']] * len(row['sighting_locations']))), axis=1)
        st.write(sightings_df[['full_text', 'sighting_locations', 'location_time']].head())

        # Visualizing Locations and Counts
        st.subheader("Location Mentions")
        all_locations = sightings_df['sighting_locations'].explode()
        location_counts = all_locations.value_counts()
        location_counts_df = location_counts.reset_index()
        location_counts_df.columns = ['Location', 'Count']

        if not location_counts_df.empty:
            plt.figure(figsize=(12, 6))
            sns.barplot(x='Count', y='Location', hue='Location', data=location_counts_df, palette='viridis', dodge=False)
            plt.title('Number of Mentions per Location for Nikol Angelova')
            plt.xlabel('Number of Mentions')
            plt.ylabel('Location')
            st.pyplot(plt)
        else:
            st.warning("No location data available to display.")

        # Temporal Pattern of Sightings
        st.write("")
        st.subheader("Temporal Pattern of Sightings")
        date_format = '%a %b %d %H:%M:%S %z %Y'
        sightings_df['date'] = pd.to_datetime(sightings_df['created_at'], format=date_format).dt.date
        daily_counts = sightings_df.groupby('date').size().reset_index(name='Count')

        if not daily_counts.empty:
            plt.figure(figsize=(12, 6))
            sns.lineplot(x='date', y='Count', data=daily_counts, marker='o')
            plt.title('Temporal Pattern of Sightings')
            plt.xlabel('Date')
            plt.ylabel('Number of Sightings')
            plt.xticks(rotation=45)
            st.pyplot(plt)
        else:
            st.warning("No sightings data available to display over time.")

        # Heatmap of Sightings Over Time by Location
        st.write("")
        st.subheader("Heatmap of Sightings Over Time by Location")
        time_location_counts_sightings = sightings_df.explode('sighting_locations').groupby(['date', 'sighting_locations']).size().reset_index(name='Count')
        time_location_pivot = time_location_counts_sightings.pivot(index='date', columns='sighting_locations', values='Count').fillna(0)

        if not time_location_pivot.empty:
            plt.figure(figsize=(16, 10))
            sns.heatmap(
                time_location_pivot,
                cmap='RdYlGn',
                linewidths=0.5,
                linecolor='white',
                annot=True,
                fmt=".0f",
                cbar_kws={'label': 'Number of Sightings'},
                square=True
            )
            plt.title('Heatmap of Sightings Over Time by Location')
            plt.xlabel('Location')
            plt.ylabel('Date')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            st.pyplot(plt)
        else:
            st.warning("No data available to generate a heatmap of sightings.")

        # Named Entity Recognition for Names
        st.write("")
        st.header("Named Entity Recognition for People/Organisation Names")

        # Function to extract person names from text including hashtags
        def extract_names_spacy(text):
            # Find all hashtags
            hashtags = re.findall(r'#\w+', text)
            
            # Split hashtags and add them to the text
            split_hashtags = ' '.join([split_camel_case(h) for h in hashtags])
            
            # Combine original text with split hashtags
            combined_text = text + ' ' + split_hashtags
            
            # Process with SpaCy NER
            doc = nlp(combined_text)
            
            # Extract names
            names = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
            
            return names

        # Apply the function to extract names
        final_df['identified_names'] = final_df['full_text'].apply(extract_names_spacy)

        # Flatten the list of names and count their occurrences
        all_names = [name for names in final_df['identified_names'] for name in names]
        name_counts = Counter(all_names)

        # Convert the counts to a DataFrame for easier plotting
        name_df = pd.DataFrame(name_counts.items(), columns=['Name', 'Count'])

        # Display the DataFrame with identified names and their counts
        st.dataframe(name_df)

        if not name_df.empty:
            plt.figure(figsize=(10, 6))
            plt.bar(name_df['Name'], name_df['Count'], color='skyblue')
            plt.xlabel('Names')
            plt.ylabel('Count')
            plt.title('Frequency of Identified Names in Tweets')
            plt.xticks(rotation=45, ha='right')
            st.pyplot(plt)
        else:
            st.warning("No names found to display.")

        # Updated Hashtag Extraction and Visualization with Time Trend
        st.write("")
        st.header("Hashtag Usage and Trends Over Time")

        def parse_hashtags(x):
            if pd.notnull(x) and x != '[]':
                try:
                    # Check if it's already a list (not a string to parse)
                    if isinstance(x, list):
                        return x
                    else:
                        return ast.literal_eval(x)
                except (ValueError, SyntaxError) as e:
                    st.warning(f"Error parsing hashtags: {e}")
                    return []
            else:
                return []

        # Step 1: Extract hashtags and timestamps
        df['entities.hashtags'] = df['entities.hashtags'].astype(str)
        df['hashtags'] = df['entities.hashtags'].apply(parse_hashtags)
        df['created_at'] = pd.to_datetime(df['created_at'], format=date_format).dt.date
        hashtag_time_pairs = [(hashtag['text'], row['created_at']) for index, row in df.iterrows() if isinstance(row['hashtags'], list) for hashtag in row['hashtags']]
        hashtag_df = pd.DataFrame(hashtag_time_pairs, columns=['hashtag', 'created_at'])
        hashtag_counts = Counter(hashtag_df['hashtag'])
        top_n = 20  # Display top 20 hashtags
        top_hashtags = hashtag_counts.most_common(top_n)
        hashtags, counts = zip(*top_hashtags)

        if len(top_hashtags) > 0:
            plt.figure(figsize=(10, 6))
            plt.barh(hashtags, counts, color='skyblue')
            plt.xlabel('Frequency')
            plt.ylabel('Hashtags')
            plt.title('Top 20 Hashtags by Frequency')
            plt.gca().invert_yaxis()  # To display the highest counts on top
            st.pyplot(plt)
        else:
            st.warning("No hashtags found to display.")

        # Step 3b: Visualize hashtag usage over time
        top_hashtag_list = [hashtag for hashtag, count in top_hashtags]
        hashtag_df_top = hashtag_df[hashtag_df['hashtag'].isin(top_hashtag_list)]
        hashtag_df_top['created_at'] = pd.to_datetime(hashtag_df_top['created_at'], format='%a %b %d %H:%M:%S %z %Y', errors='coerce')
        hashtag_time_counts = hashtag_df_top.groupby([hashtag_df_top['created_at'].dt.date, 'hashtag']).size().unstack(fill_value=0)

        if not hashtag_time_counts.empty:
            plt.figure(figsize=(15, 8))
            for hashtag in top_hashtag_list:
                plt.plot(hashtag_time_counts.index, hashtag_time_counts[hashtag], label=f'#{hashtag}')
            plt.xlabel('Date')
            plt.ylabel('Number of Occurrences')
            plt.title('Hashtag Usage Over Time')
            plt.legend(title='Hashtags')
            plt.xticks(rotation=45)
            st.pyplot(plt)
        else:
            st.warning("No hashtag data available to display over time.")

        # Keyword Analysis
        st.write("")
        st.header("Top Keywords in Tweets")

        # Preprocessing setup
        stop_words = set(nltk.corpus.stopwords.words('english'))
        punctuation = set(string.punctuation)

        def preprocess(text):
            text = text.lower()
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            text = re.sub(r'#\w+', '', text)
            text = re.sub(r'@\w+', '', text)
            tokens = nltk.word_tokenize(text)
            tokens = [word for word in tokens if word not in stop_words and word not in punctuation]
            return tokens

        tokens = relevant_tweets_df['full_text'].apply(preprocess)
        all_words = [word for tweet in tokens for word in tweet]
        word_freq = Counter(all_words)
        most_common = word_freq.most_common(20)

        if most_common:
            words, counts = zip(*most_common)
            plt.figure(figsize=(10, 6))
            plt.barh(words, counts)
            plt.xlabel('Frequency')
            plt.title('Top 20 Keywords in Tweets')
            plt.gca().invert_yaxis()  # Invert y-axis to show the highest frequency on top
            st.pyplot(plt)
        else:
            st.warning("No keywords found to display.")

        # Top Retweeted and Liked Tweets
        st.header("Top 10 Most Retweeted and Liked Tweets")

        # Filter the DataFrame to include only relevant tweets (predicted_label = 1)
        relevant_tweets = new_data[new_data['predicted_label'] == 1].copy()
        # Extract relevant columns: full_text, retweet_count, favorite_count, and entities.urls
        def extract_url(entities_urls):
            try:
                urls = json.loads(entities_urls) if isinstance(entities_urls, str) else entities_urls
                if isinstance(urls, list) and len(urls) > 0:
                    return urls[0]['expanded_url']
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                st.warning(f"Error parsing URL: {e}")
            return None

        # Extract URLs and ensure numerical columns are properly converted
        relevant_tweets['url'] = relevant_tweets['entities.urls'].apply(extract_url)
        df_relevant = relevant_tweets[['full_text', 'retweet_count', 'favorite_count', 'url']].copy()

        # Convert retweet_count and favorite_count to integers, handling any potential errors
        df_relevant['retweet_count'] = pd.to_numeric(df_relevant['retweet_count'], errors='coerce').fillna(0).astype(int)
        df_relevant['favorite_count'] = pd.to_numeric(df_relevant['favorite_count'], errors='coerce').fillna(0).astype(int)

        # Sort by retweet_count and favorite_count to get the top 10 tweets
        top_retweeted = df_relevant.sort_values(by='retweet_count', ascending=False).head(10)
        top_liked = df_relevant.sort_values(by='favorite_count', ascending=False).head(10)
        
        st.write("")
        st.subheader("Top 10 Most Retweeted Relevant Tweets")
        st.table(top_retweeted[['full_text', 'retweet_count', 'url']])

        st.write("")
        st.subheader("Top 10 Most Liked Relevant Tweets")
        st.table(top_liked[['full_text', 'favorite_count', 'url']])

        st.success("Report generated successfully!")

    except Exception as e:
        st.error(f"An error occurred: {e}")

# End of the Streamlit app
