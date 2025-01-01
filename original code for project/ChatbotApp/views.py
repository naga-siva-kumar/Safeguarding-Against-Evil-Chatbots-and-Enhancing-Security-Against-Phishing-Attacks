from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pymysql
import random

global precision, recall, fscore, accuracy, uname

X = np.load("model/X.txt.npy")
Y = np.load("model/Y.txt.npy")
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]


with open('model/tfidf.txt', 'rb') as file:
    tfidf = pickle.load(file)
file.close()
X = tfidf.fit_transform(X).toarray()
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

if os.path.exists('model/svm.txt'):
    with open('model/svm.txt', 'rb') as file:
        svm_cls = pickle.load(file)
    file.close()
else:
    svm_cls = svm.SVC()
    svm_cls.fit(X_train, y_train)
    with open('model/svm.txt', 'wb') as file:
        pickle.dump(svm_cls, file)
    file.close()

if os.path.exists('model/rf.txt'):
    with open('model/rf.txt', 'rb') as file:
        rf_cls = pickle.load(file)
    file.close()
else:
    rf_cls = RandomForestClassifier()
    rf_cls.fit(X_train, y_train)
    with open('model/rf.txt', 'wb') as file:
        pickle.dump(rf_cls, file)
    file.close()

if os.path.exists('model/dt.txt'):
    with open('model/dt.txt', 'rb') as file:
        dt_cls = pickle.load(file)
    file.close()
else:
    dt_cls = RandomForestClassifier()
    dt_cls.fit(X_train, y_train)
    with open('model/dt.txt', 'wb') as file:
        pickle.dump(dt_cls, file)
    file.close()

def RunSVM(request):
    if request.method == 'GET':
        # Initialize global variables
        global precision, recall, fscore, accuracy
        global X_train, X_test, y_train, y_test, svm_cls
        
        # Ensure these variables are initialized as lists
        precision = []
        recall = []
        fscore = []
        accuracy = []
        
        # Run predictions
        predict = svm_cls.predict(X_test)
        acc = accuracy_score(y_test, predict) * 100
        p = precision_score(y_test, predict, average='macro') * 100
        r = recall_score(y_test, predict, average='macro') * 100
        f = f1_score(y_test, predict, average='macro') * 100
        
        precision.append(p)
        recall.append(r)
        fscore.append(f)
        accuracy.append(acc)
        
        output = ""
        output += '<tr><td><font size="" color="black">Random Forest</td>'
        output += '<td><font size="" color="black">' + str(accuracy[0]) + '</td>'
        output += '<td><font size="" color="black">' + str(precision[0]) + '</td>'
        output += '<td><font size="" color="black">' + str(recall[0]) + '</td>'
        output += '<td><font size="" color="black">' + str(fscore[0]) + '</td></tr>'
    
        output += '<tr><td><font size="" color="black">Decision Tree</td>'
        output += '<td><font size="" color="black">' + str(accuracy[0]) + '</td>'
        output += '<td><font size="" color="black">' + str(precision[0]) + '</td>'
        output += '<td><font size="" color="black">' + str(recall[0]) + '</td>'
        output += '<td><font size="" color="black">' + str(fscore[0]) + '</td></tr>'
    
        output += '<tr><td><font size="" color="black">SVM</td>'
        output += '<td><font size="" color="black">' + str(accuracy[0]) + '</td>'
        output += '<td><font size="" color="black">' + str(precision[0]) + '</td>'
        output += '<td><font size="" color="black">' + str(recall[0]) + '</td>'
        output += '<td><font size="" color="black">' + str(fscore[0]) + '</td></tr>'
        
        LABELS = ['Normal URL', 'Phishing URL']
        conf_matrix = confusion_matrix(y_test, predict)
        plt.figure(figsize=(6, 6))
        ax = sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, cmap="viridis", fmt="g")
        ax.set_ylim([0, 2])
        plt.title("SVM Confusion Matrix")
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        plt.show()
        
        context = {'data': output}
        return render(request, 'ViewOutput.html', context)

def RunDT(request):
    if request.method == 'GET':
        global precision, recall, fscore, accuracy
        global X_train, X_test, y_train, y_test
        
        predict = dt_cls.predict(X_test)
        acc = accuracy_score(y_test,predict)*100
        p = precision_score(y_test,predict,average='macro') * 100
        r = recall_score(y_test,predict,average='macro') * 100
        f = f1_score(y_test,predict,average='macro') * 100
        precision.append(p)
        recall.append(r)
        fscore.append(f)
        accuracy.append(acc)
        output = ""
        output+='<tr><td><font size="" color="black">Random Forest</td>'
        output+='<td><font size="" color="black">'+str(accuracy[0])+'</td>'
        output+='<td><font size="" color="black">'+str(precision[0])+'</td>'
        output+='<td><font size="" color="black">'+str(recall[0])+'</td>'
        output+='<td><font size="" color="black">'+str(fscore[0])+'</td>'

        output+='<tr><td><font size="" color="black">Decision Tree</td>'
        output+='<td><font size="" color="black">'+str(accuracy[1])+'</td>'
        output+='<td><font size="" color="black">'+str(precision[1])+'</td>'
        output+='<td><font size="" color="black">'+str(recall[1])+'</td>'
        output+='<td><font size="" color="black">'+str(fscore[1])+'</td>'
        
        LABELS = ['Normal URL','Phishing URL']
        conf_matrix = confusion_matrix(y_test, predict) 
        plt.figure(figsize =(6, 6)) 
        ax = sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g");
        ax.set_ylim([0,2])
        plt.title("Decision Tree Confusion matrix") 
        plt.ylabel('True class') 
        plt.xlabel('Predicted class') 
        plt.show()    
        context= {'data':output}
        return render(request, 'ViewOutput.html', context)    


def RunRandom(request):
    if request.method == 'GET':
        global precision, recall, fscore, accuracy, rf_cls
        global X_train, X_test, y_train, y_test
        precision = []
        recall = []
        fscore = []
        accuracy = []
        predict = rf_cls.predict(X_test)
        acc = accuracy_score(y_test,predict)*100
        p = precision_score(y_test,predict,average='macro') * 100
        r = recall_score(y_test,predict,average='macro') * 100
        f = f1_score(y_test,predict,average='macro') * 100
        precision.append(p)
        recall.append(r)
        fscore.append(f)
        accuracy.append(acc)
        output = ""
        output+='<tr><td><font size="" color="black">Random Forest</td>'
        output+='<td><font size="" color="black">'+str(accuracy[0])+'</td>'
        output+='<td><font size="" color="black">'+str(precision[0])+'</td>'
        output+='<td><font size="" color="black">'+str(recall[0])+'</td>'
        output+='<td><font size="" color="black">'+str(fscore[0])+'</td>'
        LABELS = ['Normal URL','Phishing URL']
        conf_matrix = confusion_matrix(y_test, predict) 
        plt.figure(figsize =(6, 6)) 
        ax = sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g");
        ax.set_ylim([0,2])
        plt.title("Random Forest Confusion matrix") 
        plt.ylabel('True class') 
        plt.xlabel('Predicted class') 
        plt.show()    
        context= {'data':output}
        return render(request, 'ViewOutput.html', context)
    
def getData(arr):
    data = ""
    for i in range(len(arr)):
        arr[i] = arr[i].strip()
        if len(arr[i]) > 0:
            data += arr[i]+" "
    return data.strip()        

def PredictAction(url_input):
    global rf_cls, tfidf
    test = []
    arr = url_input.split("/")
    output = "Unable to predict"
    print('url',url_input)
    if len(arr) > 0:
        data = getData(arr)
        #print(data)
        test.append(data)
        test = tfidf.transform(test).toarray()
        #print(test)
        #print(test.shape)
        predict = rf_cls.predict(test)
        #print(predict)
        predict = predict[0]
        if predict == 0:
            output = "Genuine URL"
        if predict == 1:
            output = "Contains PHISHING URL"
    return output       

# def getAccountNo(question):
#     acc = "none"
#     arr = question.split(" ")
#     for i in range(len(arr)):
#         try:
#             acc = int(arr[i].strip())
#             break
#         except Exception:
#             pass
#     return acc

import pymysql

# def getBalance(acc_no):
#     output = "Invalid account number"
    
#     # Database connection
#     try:
#         con = pymysql.connect(
#             host='127.0.0.1', 
#             port=3306, 
#             user='root', 
#             password='naga123@.com', 
#             database='EvilChatbot', 
#             charset='utf8'
#         )
        
#         # Initialize amounts
#         deposit = 0
#         debit = 0
        
#         # Use context manager to automatically close the cursor
#         with con:
#             with con.cursor() as cur:
#                 # Fetch the sum of deposits
#                 cur.execute("SELECT SUM(transaction_amount) FROM account WHERE acc_no=%s AND transaction_type='deposit'", (acc_no,))
#                 row = cur.fetchone()
#                 if row and row[0] is not None:
#                     deposit = float(row[0])
            
#             with con.cursor() as cur:
#                 # Fetch the sum of debits
#                 cur.execute("SELECT SUM(transaction_amount) FROM account WHERE acc_no=%s AND transaction_type='debit'", (acc_no,))
#                 row = cur.fetchone()
#                 if row and row[0] is not None:
#                     debit = float(row[0])
        
#         # Calculate balance
#         if deposit > 0:
#             balance = deposit - debit
#             output = f"Your current balance is: {balance:.2f}"
    
#     except pymysql.MySQLError as e:
#         # Handle any errors that occur during the database connection or query execution
#         output = f"Error fetching balance: {e}"
    
#     finally:
#         # Ensure the connection is closed
#         if con and con.open:
#             con.close()
    
#     return output
# def getDebit(acc_no):
#     output = "Transaction No, Transaction Type, Transaction Amount, Balance, Transaction Date\n"
#     con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'naga123@.com', database = 'EvilChatbot',charset='utf8')
#     with con:
#         cur = con.cursor()
#         cur.execute("select * from account where acc_no='"+str(acc_no)+"' and transaction_type='debit'")
#         rows = cur.fetchall()
#         for row in rows:
#             output += row[0]+", "+row[1]+", "+row[2]+", "+row[3]+", "+row[4]+"\n"
#     return output

# def getDeposit(acc_no):
#     output = "Transaction No, Transaction Type, Transaction Amount, Balance, Transaction Date\n"
#     con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'naga123@.com', database = 'EvilChatbot',charset='utf8')
#     with con:
#         cur = con.cursor()
#         cur.execute("select * from account where acc_no='"+str(acc_no)+"' and transaction_type='deposit'")
#         rows = cur.fetchall()
#         for row in rows:
#             output += row[0]+", "+row[1]+", "+row[2]+", "+row[3]+", "+row[4]+"\n"
    # return output

def ChatData(request):
    if request.method == 'GET':
        global email
        links= ['http://www.google.com', 'http://mail.google.com', 'aws.amazon.com', 'http://www.yahoo.com', 'http://allrecipes.com/video/466/parmesan-cheese-twists/detail.aspx?prop24=VH_Brands',
                'http://askubuntu.com/questions/239450/nvidia-7800-gtx-drivers-not-working-for-12-04', 'http://bestblackhatforum.com/Thread-Become-VIP-Is-Your-Best-Bet-Why?action=lastpost']
        
        question = request.GET.get('mytext', False)
        link =question[:4]
        if (link =='http'):
            malicious_output = PredictAction(question)
            output = malicious_output
            link=question
            return HttpResponse("Chatbot: "+output+"#"+link+"#"+malicious_output, content_type="text/plain") 
        else:
            question = question.strip("\n").strip()
            question = question.lower()
            # acc_no = getAccountNo(question)
            # print(str(acc_no)+" ====")
            # output = "not found"
            # if "balance" in question:
            #     output = getBalance(acc_no)
            # if "debit" in question:
            #     output = getDebit(acc_no)
            # if "deposit" in question:
            #     output = getDeposit(acc_no)
            return HttpResponse("Chatbot: "+output+"#", content_type="text/plain")            
        

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def Predict(request):
    if request.method == 'GET':
       return render(request, 'Predict.html', {})
    
def UserLogin(request):
    if request.method == 'GET':
        return render(request, 'UserLogin.html', {})

def Chatbot(request):
    if request.method == 'GET':
        return render(request, 'Chatbot.html', {})    

def UserLoginAction(request):
    if request.method == 'POST':
        global uname
        user = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        index = 0
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'naga123@.com', database = 'EvilChatbot',charset='utf8')
        with con:    
            cur = con.cursor()
            cur.execute("select * FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == user and password == row[1]:
                    uname = user
                    index = 1
                    break		
        if index == 1:
            context= {'data':'Welcome '+user}
            return render(request, 'UserScreen.html', context)
        else:
            context= {'data':'Invalid Login'}
            return render(request, 'UserLogin.html', context)

def Register(request):
    if request.method == 'GET':
        return render(request, 'Register.html', {})    

def RegisterAction(request):
    if request.method == 'POST':
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        email = request.POST.get('t4', False)
        address = request.POST.get('t5', False)
        
        status = "none"
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'naga123@.com', database = 'EvilChatbot',charset='utf8')
        with con:    
            cur = con.cursor()
            cur.execute("select username FROM register where username='"+username+"'")
            rows = cur.fetchall()
            for row in rows:
                status = "Username Already Exists"
                break
        if status == "none":
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'naga123@.com', database = 'EvilChatbot',charset='utf8')
            db_cursor = db_connection.cursor()
            student_sql_query = "INSERT INTO register(username,password,contact,email,address) VALUES('"+username+"','"+password+"','"+contact+"','"+email+"','"+address+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                status = 'Signup Process Completed'
        context= {'data': status}
        return render(request, 'Register.html', context)

