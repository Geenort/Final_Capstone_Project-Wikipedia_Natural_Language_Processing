from flask import Flask, request, render_template, session, redirect
import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
nlp = spacy.load('en_core_web_sm')

# Similarity Processing

file_one = "Resources/MoviePlots.csv"

df = pd.read_csv(file_one)

userplot_raw = "Teenager Miles Morales struggles to live up to the expectations of his father, police officer Jefferson Davis, who sees Spider-Man as a menace. Miles transfers to a boarding school, but later sneaks out and goes to his uncle Aaron Davis's house. When he takes Miles to an abandoned subway station to paint graffiti, Miles is bitten by a radioactive spider and gains spider-like abilities."

userplot = []    
doc = nlp(userplot_raw)
tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
userplot.append(tokens)

for i in range (len(userplot)):
    userplot = " ".join(userplot[0])

newdf = df[~df.Plot.isna()]

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(newdf.NoStopwords)

newdf['tfidf'] = tfidf

similarity = tfidf[0]*vectorizer.transform([userplot]).T

sims = []
for i in range (len(newdf)):
    similarity = tfidf[i]*vectorizer.transform([userplot]).T
    sims.append(similarity[0,0])
newdf['Similarities'] = sims

topfive_full = newdf.sort_values(by='Similarities', ascending=False).head(5)

topfive = topfive_full[['Release Year', 'Title', 'Director', 'Genre', 'Plot', 'Similarities']]

# Display Top 5 Dataframe

app = Flask(__name__)


@app.route('/', methods=("POST", "GET"))
def html_table():

    return render_template('simple.html',  tables=[topfive.to_html(classes='data', header="true")])


if __name__ == '__main__':
    app.run(debug=True)