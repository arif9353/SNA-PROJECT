from flask import Flask, render_template, request, session, redirect, url_for
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import os
import logging
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
from flask_session import Session



app = Flask(__name__)
app.config['SECRET_KEY'] = 'arif'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)
LANGUAGE = "english"


@app.route('/')
def home():
    return render_template("opening.html")

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
# Set TF_ENABLE_ONEDNN_OPTS to 0 to suppress related warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')


def sentiment_score(review):
    tokens =  tokenizer.encode(review, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits))+1


def sentiment_comments(urlll):
    link = urlll
    data = []
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--headless")  # If you want to run Chrome in headless mode 
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("start-maximized")
    chrome_options.add_argument("disable-infobars")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--no-sandbox")
    # Set the path to the chromedriver.exe file
    chrome_options.binary_location = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    # Initialize the Chrome driver with chrome_options
    driver = webdriver.Chrome(options=chrome_options)
    with driver:
        wait = WebDriverWait(driver, 15)
        driver.get(f"{link}")
        for _ in range(12):  # You might need to adjust the number of scrolls
            driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(5)

        for comment in wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#content-text"))):
            data.append(comment.text)
    print(data)
    import pandas as pd   
    frame= pd.DataFrame(data, columns=['comment'])
    frame = pd.DataFrame(np.array(data), columns=['review'])
    frame['sentiment'] = frame['review'].apply(lambda x: sentiment_score(x))
    average_sentiment =frame['sentiment'].mean()
    comentzz = data[:10]
    answer = []
    answer.append(average_sentiment)
    answer.append(comentzz)
    return answer



@app.route('/youtube', methods=['GET', 'POST'])
def youtube():
    if request.method == 'POST':
        urll = request.form.get('url_youtube')
        if urll:
            ans = sentiment_comments(urll)
            print(ans[0])
            print('\n\n\n')
            print(ans[1])
            session['sentiments'] = ans[0]  
            session['comments'] = ans[1]
        return redirect(url_for('final_youtube'))
    return render_template("youtube.html")


@app.route('/final_youtube', methods=['GET','POST'])
def final_youtube():
    sentiment = session['sentiments']
    senti =  round(sentiment,2) 
    comment = session['comments']
    return render_template("final_youtube.html",rating=senti, keyword=comment)



if __name__ == '__main__':
    app.run(debug=True)