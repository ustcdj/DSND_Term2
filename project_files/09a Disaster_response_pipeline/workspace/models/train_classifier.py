import sys
import re
import numpy as np
import pandas as pd
import pickle

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.base import BaseEstimator, TransformerMixin

from utility import tokenize, StartingVerbExtractor, TextLenghExtractor
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score


def load_data(database_filepath):
    engine = create_engine("sqlite:///" + database_filepath)
    df = pd.read_sql_table('messages', con=engine)

    X = df['message']
    y = df.iloc[:, 4:]
    y['related'].replace(2, y['related'].mode().iloc[0], inplace=True)
    category_names = list(y.columns)

    X = X.head(1000)
    y = y.head(1000)

    return X, y, category_names

'''
def tokenize(text):

    # Extract all the urls from the provided text
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)

    # Replace url with a url placeholder string
    for url in detected_urls:
        text = text.replace(url, urlplaceholder )

    # didn't remove punctuation, seems to cause error 'list out of range'
    # text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # normalize text
    tokens = word_tokenize(text)

    # didn't remove stop words. slow down the process, also decreases model performance metrics
    # kokens = [w for w in tokens if w not in stopwords.words("english")]

    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


class TextLenghExtractor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.Series(X).apply(lambda x: len(x))
        return pd.DataFrame(X)
'''

def build_model():

    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor()),
            ('text_length', TextLenghExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100)))
    ])

    parameters = {
    #    'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
    #    'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
    #    'features__text_pipeline__vect__max_features': (None, 5000, 10000),
    #    'features__text_pipeline__tfidf__use_idf': (True, False),
    #    'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 3],

        'features__transformer_weights': (
            {'text_pipeline': 1, 'starting_verb': 0.5, 'text_length': 0.5},
            {'text_pipeline': 1, 'starting_verb': 0.5, 'text_length': 1},
            {'text_pipeline': 1, 'starting_verb': 1, 'text_length': 1},
        )
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2, n_jobs=2, verbose=2)

    return cv

def evaluate_model(model, X_test, y_test, category_names):

    y_pred = model.predict(X_test)

    # Overall scores
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='micro')
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    print("Overall model performance (tuned)")
    print("-"*27)
    print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}")


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
