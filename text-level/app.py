from flask import Flask, render_template, url_for, request, jsonify
import os
from sklearn.feature_extraction.text import CountVectorizer
import string
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
import re
import math
from sklearn.manifold import TSNE
import numpy as np
from gensim.models import KeyedVectors
from asyncio.windows_events import NULL
from datetime import datetime
from re import I
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import pythainlp
from pythainlp import word_tokenize
from pythainlp.util import dict_trie
from sklearn import preprocessing
le = preprocessing.LabelEncoder()


os.getcwd
# os.chdir('C:\\Users\\Admin\\Downloads\\TextLevel\\frontend')
os.getcwd

"""## Train """

 # เรียกใช้ไฟล์ train.csv
df = pd.read_csv(r'train.csv')
datatrain = pd.DataFrame(df, columns=['text', 'labels'])                     
train_text = datatrain['text']                            
train_label = datatrain['labels']




# แบ่งข้อมูลไว้เทส67.88% เทรน32.12%  , แรนด้อมคำทุก1412
train_df, valid_df = train_test_split(df, test_size=0.6788, random_state=1412)


train = df[train_text != "#ERROR!"].reset_index(drop=True)                                


#ฟังก์ชันตัดคำ 
def split_word(text):

    #ใช้ pythainlp engine newmm
    tokens = word_tokenize(text, engine='newmm')

    # ลบตัวเลข
    tokens = [i for i in tokens if not i.isnumeric()]

    # ลบช่องว่าง
    tokens = [i for i in tokens if not ' ' in i]

    return tokens

#ตัด emoji
def deEmojify(text):
    regrex_pattern = re.compile(pattern="["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                # transport & map symbols                            #สร้างฟังชั่นเคลียอิโมจิ
                                u"\U0001F680-\U0001F6FF"
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                "]+", flags=re.UNICODE)
    return regrex_pattern.sub(r'', text)


# ทำการสร้างฟังชั่นก์ clean_msg
def clean_msg(msg):

    # ลบ text ที่อยู่ในวงเล็บ <> ทั้งหมด
    msg = re.sub(r'<.*?>', '', msg)

    # ลบ hashtag
    msg = re.sub(r'#', '', msg)

    # ลบ เครื่องหมายคำพูด (punctuation)
    for c in string.punctuation:
        msg = re.sub(r'\{}'.format(c), '', msg)

    # ลบ separator เช่น \n \t
    msg = ' '.join(msg.split())

    return msg


#สร้างตัวแปร new_train_df ทำการ clean msg datafram train_text และนำมาตัดคำทีละประโยค
new_train_df = [split_word(i) for i in [(i)for i in [clean_msg(i) for i in train_text]]]

# เขียน loop นับจำนวนประโยค i ใน tokenize_train_df
df_list = []
for i in new_train_df:
    tokens_list_train = new_train_df
    for j in i:
        if j not in df_list:
            df_list.append(j)


def Sent(sent):  # รับคำมา ละรีเทิร์นคำกลับ
    return sent


# สร้างตัวแปร cvec รับค่า CountVectorizer
cvec = CountVectorizer(analyzer='word', tokenizer=Sent,
                       vocabulary=df_list, lowercase=False)
c_feat = cvec.fit_transform(new_train_df)

# นำ new_train_df มาทำ countvec โดยใช้ fit_transform
cf_train_Sent = cvec.fit_transform(new_train_df)
trainset = cf_train_Sent

#นำ label มา fit_transform
targets = le.fit_transform(train_label)

#ทำการแบ่ง train test ด้วย train_test_split ของ sklearn โดยแบ่ง test size 25% ของข้อมูล และสลับข้อมูลทุกๆ 5 ประโยค
X_train, X_test, Y_train, Y_test = train_test_split(
    trainset, targets, test_size=0.25, random_state=5)

model = MultinomialNB()                                         #เรียกใช้โมเดล MultinomialNB
MNB = model.fit(X_train, Y_train)

#ฟังก์ชัน ทำนายข้อความและแสดงผลลัพธ์เป็น pos neg โดย 0 เป็น neg , 1 เป็น pos 
def pred(text):
    tokenize_word = split_word(clean_msg(text))                      #   

    cf_test_Sent = cvec.fit_transform(tokenize_word)
    testword = cf_test_Sent

    my_predictions = MNB.predict(testword)

    result_predict = ""
    for check in my_predictions:
        if check == 0:
            result_predict = "neg"
        elif check == 1:
            result_predict = "pos"
    return result_predict


#นับจำนวนคำ ของประโยคที่ input
def lenword(text):
    len_word = len(split_word(clean_msg(text)))
    return len_word

#ตัดคำที่ input
def tokenize(text):
    tokenize_txt = (' '.join(split_word(clean_msg(text))))
    return tokenize_txt

#ความแม่นยำ neg 
def acc_f(text):
    tokenize_test_Sent = [split_word(clean_msg(text))]
    cf_test_Sent = cvec.fit_transform(tokenize_test_Sent)
    testsent = cf_test_Sent
    pred_proba = model.predict_proba(testsent)

    allproba = []
    final_prob = []
    for prob in pred_proba:
        allproba = prob

    if allproba[0] > allproba[1]:
        final_prob = allproba[0]
    else:

        final_prob = allproba[1]

    return "{0:.2f}".format(final_prob)

#ความแม่นยำ pos
def acc_t(text):
    tokenize_test_Sent = [split_word(clean_msg(text))]
    cf_test_Sent = cvec.fit_transform(tokenize_test_Sent)
    testsent = cf_test_Sent
    pred_proba = model.predict_proba(testsent)

    allproba = []
    for prob in pred_proba:
        allproba = prob

    return "{0:.2f}".format(allproba[1])

#จับรวม tranform text  นำไปpredict  แล้วนำเข้า result
def etl(mess):
    tokenize_txt = tokenize(mess)
    Prediction = pred(mess)
    len_word = lenword(mess)
    Acc_f = acc_f(mess)
    Acc_t = acc_t(mess)
    return {"message": mess, "tokenize_txt": tokenize_txt, "pred": Prediction, "len_word": len_word, "acc_f": Acc_f, "acc_t": Acc_t}


app = Flask(__name__)
app.debug = True


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/result', methods=['POST'])
def result():
    mess = request.form['message']
    phrases = mess.split("\r\n")
    print(f'phrases: {phrases}')

    results = []
    for it in phrases:
        results.append(etl(it)) 

    print(results)
    # result = etl(mess)
    # print(f'result: {result}')
    pos = 0 
    neg = 0
    for it in results:
        if it["pred"] == "pos":
            pos += 1 #it.acc_f
        else:
            neg += 1 #it.acc_f
            

    return render_template("result.html", results=results, pos=(pos/len(results)*100), neg=(neg/len(results)*100) )

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
