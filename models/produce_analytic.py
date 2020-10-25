#import library
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import gensim.summarization

#load data
def load_data():
    database_filepath = 'sqlite:///../data/DisasterResponse.db'
    engine = create_engine(database_filepath)
    df = pd.read_sql('SELECT * FROM clean_data' , engine)
    text = df['message'].str.cat(sep='... ')
    text = text[:50000]
    return text

#produce word cloud
def produce_wordcloud(text):
    stopwords = STOPWORDS.union({'need','one','will','said','thank','people','now','help','area','well','including', 'us', 'information','know','work','time','day','government','country','many','please','much','service','region','new'})
    wordcloud = WordCloud(stopwords=stopwords,background_color='white',width=7000,height=7000,max_words=400).generate(text)
    fig = plt.figure(figsize=(20,10))
    plt.imshow(wordcloud)
    plt.axis('off')
    fig.savefig('world_cloud.png', dpi = 150)

def produce_gensim_summary(text):
    #gensim summarization
    summary = gensim.summarization.summarize(text, ratio=0.0001) 
    f = open("gensim_summary.txt", "w")
    f.write(summary)
    f.close()

def main():
    print("loading data...")
    text = load_data()
    
    print("producing word cloud...")
    produce_wordcloud(text)
    
    print("producing gensim text summary...")
    produce_gensim_summary(text)
    
    print('word cloud & gensim summary saved!')

if __name__ == '__main__':
    main()
