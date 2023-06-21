''' Functions - Import CSV API, Preprocessing Textual Data Api, ML Model Selection API, '''

from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
import threading
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, recall_score, precision_score
from sklearn.model_selection import train_test_split
import time
import nltk
nltk.download('punkt')
nltk.download('wordnet')
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string, re
import pickle


app = Flask(__name__, static_folder='static')

'''Global variables'''
df_train = None
df_test = None
training_size = None
trim_rows = None
df_train_trimmed = None
selected_columns = None
df_filtered_trimmed = None
text_columns = []
f1_score_lg = []
f1_score_nb = []
f1_score_rf = []
f1_score_svc = []
f1_score_dt = []

recall_score_lg = []
recall_score_nb = []
recall_score_rf = []
recall_score_svc = []
recall_score_dt = []

precision_score_lg = []
precision_score_nb = []
precision_score_rf = []
precision_score_svc = []
precision_score_dt = []
global f1score

'''Preprocessing functions'''
def preprocess_text(text):
  if isinstance(text, str):

    """Remove URLs from a sample string"""
    text =  re.sub(r"http\S+", "", text)
    text = re.sub(r"[\[\],@\'?\.$%_:()\-\"&;<>{}|+!*#]", " ", text, flags=re.I)
    text = ' '.join(w for w in text.split() if not any(x.isdigit() for x in w)) 
    text = text.lower()
    
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = ' '.join([lemmatizer.lemmatize(w) for w in text])
    
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    text = text.split() 
    text = [t for t in text if not t.isdigit()] 
    text = " ".join(text) 

    '''Tokenize the text'''
    tokens = word_tokenize(text.lower())

    # '''Remove punctuation'''
    tokens = [token for token in tokens if token not in string.punctuation]

    # '''Remove extra white spaces'''
    tokens = [token.strip() for token in tokens if token.strip()]

    # 
    text = text.lower()
    return text
  else:
        return str(text)



def delete_image_files(file_paths, delay):
    """
    Delete the specified image files after a certain delay.
    """
    def delete_files():
        time.sleep(delay)
        for file_path in file_paths:
            os.remove(file_path)

    deletion_thread = threading.Thread(target=delete_files)
    deletion_thread.start()

    
''' Function - Handles the upload of the train data CSV file
        Pass In: train data (.csv)
        Pass Out: train data, graph, column and rows count, features
    Endfunction '''
@app.route('/upload/train_data', methods=['POST'])
def upload_train_data():
    global df_train
    global training_size
    file = request.files['file']
    training_size = float(request.form.get('training_size'))
    if not file:
        return jsonify({'error': 'No file provided'})

    try:
        df_train = pd.read_csv(io.StringIO(file.read().decode('utf-8')))
        csv_path = os.path.join(app.static_folder, 'data.csv')
        df_train.to_csv(csv_path, index=False)
        rows, columns = df_train.shape
        
        return jsonify({'success': True,
                        'df_train':df_train.to_dict(), 
                        'rows': rows, 
                        'columns': columns,
                        'training_size': training_size
                        })
    except Exception as e:
        return jsonify({'error': str(e)}) 
    

''' Function - Handles the upload of the test data CSV file
        Pass In: test data (.csv)
        Pass Out: test data, graph, column and rows count, features
    Endfunction '''  
@app.route('/upload/test_data', methods=['POST'])
def upload_test_data():
    global df_test
    file1 = request.files['file']
    if not file1:
        return jsonify({'error': 'No file provided'})

    try:
        df_test = pd.read_csv(io.StringIO(file1.read().decode('utf-8')))
        rows, columns = df_test.shape
        return jsonify({'success': True, 
                        'rows': rows, 
                        'columns': columns, 
                        })
    except Exception as e:
        return jsonify({'error': str(e)})


''' Function -  serves static files from the server's static folder.
        Pass In: any data
        Pass Out: new data added in static folder with path
    Endfunction '''
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory(app.static_folder, path)


''' Function -  filters columns based on user input in Multiselection Bar
        Pass In: required columns
        Pass Out: selected columns array
    Endfunction '''
@app.route('/filter_columns', methods=['POST'])
def filter_columns():
    global df_train
    global selected_columns
    global df_test

    try:
        selected_columns = request.json['columns']
        selected_columns = [column for column in selected_columns]
        df_train = df_train[selected_columns]
        return jsonify({'success': True, 'selected_columns': selected_columns})

    except Exception as e:
        return jsonify({'error': str(e)})


''' Function - trims data as per training size
        Pass In: dataframe
        Pass Out: preprocessed and trimmed dataframe
    Endfunction '''
@app.route('/trim_data', methods=['POST'])
def trim_data():
    global df_train
    global trim_rows
    global df_train_trimmed
    global training_size
    global selected_columns
    global df_filtered_trimmed

    csv = os.path.join(app.static_folder, 'filtered_trimmed.csv')
    rows, columns = df_train.shape
    trim_rows = int(training_size * rows)
    df_train['req1'] = df_train['req1'].apply(preprocess_text)
    df_train['req2'] = df_train['req2'].apply(preprocess_text)

    df_train_trimmed = df_train[:trim_rows]
    df_filtered_trimmed = df_train_trimmed[selected_columns]
    df_filtered_trimmed.to_csv(csv, index=False)

    for column in df_filtered_trimmed.columns:
        if df_filtered_trimmed[column].dtype == object or isinstance(df_filtered_trimmed[column].dtype, pd.StringDtype):
            text_columns.append(column)
    return jsonify({'success': True,
                    'df_filtered_trimmed' : df_filtered_trimmed, 'text_columns': text_columns
                     })


# ML models - STEP 2
''' Following pseudocode remains same for all ML models
    Function - Logistic Regression / Naive Bayes / Support Vector Machine / Random Forest / Decision Tree
            Pass In: dataframe (both train + test)
            Pass Out: accuracy, graphs, f1 scores, recall scores, precision score, predicted values in JSON
    Endfunction '''


@app.route('/logistic-regression', methods=['POST'])
def perform_logistic_regression():
    global df_test
    global df_filtered_trimmed
    global f1_score_lg
    global recall_score_lg
    global precision_score_lg
    recall_score_lg.clear()
    precision_score_lg.clear()
    f1_score_lg.clear()
    

    try:
        start = time.time()
        X = df_filtered_trimmed.drop(columns=['Label'])
        y = df_filtered_trimmed['Label']
        X_text = X[text_columns].apply(lambda x: ' '.join(x.dropna()), axis=1)

        # Load the logistic regression model from pickle file
        with open('logistic_regression.pkl', 'rb') as file:
            model = pickle.load(file)
        # Load the vectorizer from pickle file
        with open('vectorizer_fn.pkl', 'rb') as file:
            vectorizer = pickle.load(file)

        # Transform the test data using the vectorizer
        X_test_vectorized = vectorizer.transform(X_text)

        # Split the dataset into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_test_vectorized, y, train_size=0.8, random_state=42)

        # Predict on the test set
        y_pred = model.predict(X_test)
        
        stop = round(time.time() - start,4)
        accuracy = round(accuracy_score(y_test,y_pred),4)*100 
        accuracy = str(accuracy)+"%"
        report = classification_report(y_test, y_pred,  zero_division=1)
        cm=confusion_matrix(y_pred,y_test)
        f1 = round(f1_score(y_test,y_pred,average='macro'),3)
      
        acc=[]
        size=[]
        
        for k in range(2,10):
            k = k/10
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = k, random_state=42)
            X_train_text = X_train[text_columns].apply(lambda x: ' '.join(x.dropna()), axis=1)
            X_test_text = X_test[text_columns].apply(lambda x: ' '.join(x.dropna()), axis=1)
            X_train_vectorized = vectorizer.fit_transform(X_train_text)
            X_test_vectorized = vectorizer.transform(X_test_text)
            model.fit(X_train_vectorized, y_train)
            y_pred = model.predict(X_test_vectorized)
            size.append(k)
            acc.append(round(accuracy_score(y_test,y_pred),2))
            f1_score_lg.append(round(f1_score(y_test,y_pred,average='macro'),2))
            recall_score_lg.append(round(recall_score(y_test,y_pred,average='macro'),2))
            precision_score_lg.append(round(precision_score(y_test,y_pred,average='macro'),2))
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        pastel_blue_palette = sns.color_palette(["#D4FAFA", "#AFD5F0", "#AFD5F0"])
        sns.heatmap(cm, annot=True, fmt='d', cmap=pastel_blue_palette, cbar=False)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title('Confusion Matrix')
        cm = os.path.join(app.static_folder,'lg_cm.png')
        plt.show()
        plt.savefig(cm)
        plt.close()

        fig = plt.figure(figsize=(5.5, 4))
        fig.patch.set_facecolor('white') 
        plt.plot(size,acc, color='#AFD5F0')
        plt.xlabel("Training Size",color='black')
        plt.ylabel("Validation Accuracy",color='black')
        lg_f1_score = os.path.join(app.static_folder, 'lg.png')
        plt.show()
        plt.savefig(lg_f1_score)
        plt.close()

        image_files_to_delete = [cm, lg_f1_score]
        delete_image_files(image_files_to_delete, delay=7)
        
        return jsonify({'success': True, 'report': report, 'accuracy':accuracy,'stop':stop,'graph':'/static/lg.png','cm':'/static/lg_cm.png','f1':f1,'f1score':f1_score_lg,'recallscore':recall_score_lg,'precisionscore':precision_score_lg})
    
    except Exception as e:
        return jsonify({'error': str(e)})

    

@app.route('/naive-bayes', methods=['POST'])
def perform_naive_bayes():
    global df_test
    global df_filtered_trimmed
    global f1_score_nb
    global recall_score_nb
    global precision_score_nb
    recall_score_nb.clear()
    precision_score_nb.clear()
    f1_score_nb.clear()

    try:
        start = time.time()
        X = df_filtered_trimmed.drop(columns=['Label'])
        y = df_filtered_trimmed['Label']
        X_text = X[text_columns].apply(lambda x: ' '.join(x.dropna()), axis=1)

        # Load the logistic regression model from pickle file
        with open('naive_bayes.pkl', 'rb') as file:
            model = pickle.load(file)
        # Load the vectorizer from pickle file
        with open('vectorizer_fn.pkl', 'rb') as file:
            vectorizer = pickle.load(file)

        # Transform the test data using the vectorizer
        X_test_vectorized = vectorizer.transform(X_text)

        # Split the dataset into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_test_vectorized, y, train_size=0.8, random_state=42)
        y_pred = model.predict(X_test)

        stop = round(time.time() - start,4)
        accuracy = round(accuracy_score(y_test,y_pred),4)*100 
        accuracy = str(accuracy)+"%"
        report = classification_report(y_test, y_pred,  zero_division=1)
        cm=confusion_matrix(y_pred,y_test)
        f1 = round(f1_score(y_test,y_pred,average='macro'),3)
      
        acc=[]
        size=[]
        
        for k in range(2,10):
            k = k/10
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = k, random_state=42)
            X_train_text = X_train[text_columns].apply(lambda x: ' '.join(x.dropna()), axis=1)
            X_test_text = X_test[text_columns].apply(lambda x: ' '.join(x.dropna()), axis=1)
            X_train_vectorized = vectorizer.fit_transform(X_train_text)
            X_test_vectorized = vectorizer.transform(X_test_text)
            model.fit(X_train_vectorized, y_train)
            y_pred = model.predict(X_test_vectorized)
            size.append(k)
            acc.append(round(accuracy_score(y_test,y_pred),2))
            f1_score_nb.append(round(f1_score(y_test,y_pred,average='macro'),2))
            recall_score_nb.append(round(recall_score(y_test,y_pred,average='macro'),2))
            precision_score_nb.append(round(precision_score(y_test,y_pred,average='macro'),2))
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        pastel_blue_palette = sns.color_palette(["#D4FAFA", "#AFD5F0", "#AFD5F0"])
        sns.heatmap(cm, annot=True, fmt='d', cmap=pastel_blue_palette, cbar=False)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title('Confusion Matrix')
        cm = os.path.join(app.static_folder,'nb_cm.png')
        plt.show()
        plt.savefig(cm)
        plt.close()

        fig = plt.figure(figsize=(5.5, 4))
        fig.patch.set_facecolor('white') 
        plt.plot(size,acc, color='#AFD5F0')
        plt.xlabel("Training Size",color='black')
        plt.ylabel("Validation Accuracy",color='black')
        nb_f1_score = os.path.join(app.static_folder, 'nb.png')
        plt.show()
        plt.savefig(nb_f1_score)
        plt.close()

        image_files_to_delete = [cm, nb_f1_score]
        delete_image_files(image_files_to_delete, delay=7)
        
        return jsonify({'success': True, 'report': report, 'accuracy':accuracy,'stop':stop,'graph':'/static/nb.png','cm':'/static/nb_cm.png','f1':f1,'f1score':f1_score_nb,'recallscore':recall_score_nb,'precisionscore':precision_score_nb})
    
    except Exception as e:
        return jsonify({'error': str(e)})

    

@app.route('/random-forest', methods=['POST'])
def perform_random_forest():
    global df_test
    global df_filtered_trimmed
    global f1_score_rf
    global recall_score_rf
    global precision_score_rf
    recall_score_rf.clear()
    precision_score_rf.clear()
    f1_score_rf.clear()

    try:
        start = time.time()
        X = df_filtered_trimmed.drop(columns=['Label'])
        y = df_filtered_trimmed['Label']
        X_text = X[text_columns].apply(lambda x: ' '.join(x.dropna()), axis=1)

        # Load the logistic regression model from pickle file
        with open('random_forest.pkl', 'rb') as file:
            model = pickle.load(file)
        # Load the vectorizer from pickle file
        with open('vectorizer_fn.pkl', 'rb') as file:
            vectorizer = pickle.load(file)

        # Transform the test data using the vectorizer
        X_test_vectorized = vectorizer.transform(X_text)

        # Split the dataset into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_test_vectorized, y, train_size=0.8, random_state=42)
        y_pred = model.predict(X_test)

        stop = round(time.time() - start,4)
        accuracy = round(accuracy_score(y_test,y_pred),4)*100 
        accuracy = str(accuracy)+"%"
        report = classification_report(y_test, y_pred,  zero_division=1)
        cm=confusion_matrix(y_pred,y_test)
        f1 = round(f1_score(y_test,y_pred,average='macro'),3)
      
        acc=[]
        size=[]
        
        for k in range(2,10):
            k = k/10
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = k, random_state=42)
            X_train_text = X_train[text_columns].apply(lambda x: ' '.join(x.dropna()), axis=1)
            X_test_text = X_test[text_columns].apply(lambda x: ' '.join(x.dropna()), axis=1)
            X_train_vectorized = vectorizer.fit_transform(X_train_text)
            X_test_vectorized = vectorizer.transform(X_test_text)
            model.fit(X_train_vectorized, y_train)
            y_pred = model.predict(X_test_vectorized)
            size.append(k)
            acc.append(round(accuracy_score(y_test,y_pred),2))
            f1_score_rf.append(round(f1_score(y_test,y_pred,average='macro'),2))
            recall_score_rf.append(round(recall_score(y_test,y_pred,average='macro'),2))
            precision_score_rf.append(round(precision_score(y_test,y_pred,average='macro'),2))
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        pastel_blue_palette = sns.color_palette(["#D4FAFA", "#AFD5F0", "#AFD5F0"])
        sns.heatmap(cm, annot=True, fmt='d', cmap=pastel_blue_palette, cbar=False)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title('Confusion Matrix')
        cm = os.path.join(app.static_folder,'rf_cm.png')
        plt.show()
        plt.savefig(cm)
        plt.close()

        fig = plt.figure(figsize=(5.5, 4))
        fig.patch.set_facecolor('white') 
        plt.plot(size,acc, color='#AFD5F0')
        plt.xlabel("Training Size",color='black')
        plt.ylabel("Validation Accuracy",color='black')
        rf_f1_score = os.path.join(app.static_folder, 'rf.png')
        plt.show()
        plt.savefig(rf_f1_score)
        plt.close()

        image_files_to_delete = [cm, rf_f1_score]
        delete_image_files(image_files_to_delete, delay=7)
        
        return jsonify({'success': True, 'report': report, 'accuracy':accuracy,'stop':stop,'graph':'/static/rf.png','cm':'/static/rf_cm.png','f1':f1,'f1score':f1_score_rf,'recallscore':recall_score_rf,'precisionscore':precision_score_rf})
    
    except Exception as e:
        return jsonify({'error': str(e)})



@app.route('/support-vector-machine', methods=['POST'])
def perform_support_vector_machine():
    global df_test
    global df_filtered_trimmed
    global f1_score_svc
    global recall_score_svc
    global precision_score_svc
    recall_score_svc.clear()
    precision_score_svc.clear()
    f1_score_svc.clear()

    try:
        start = time.time()
        X = df_filtered_trimmed.drop(columns=['Label'])
        y = df_filtered_trimmed['Label']
        X_text = X[text_columns].apply(lambda x: ' '.join(x.dropna()), axis=1)

        # Load the logistic regression model from pickle file
        with open('svc.pkl', 'rb') as file:
            model = pickle.load(file)
        # Load the vectorizer from pickle file
        with open('vectorizer_fn.pkl', 'rb') as file:
            vectorizer = pickle.load(file)

        # Transform the test data using the vectorizer
        X_test_vectorized = vectorizer.transform(X_text)

        # Split the dataset into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_test_vectorized, y, train_size=0.8, random_state=42)
        y_pred = model.predict(X_test)

        stop = round(time.time() - start,4)
        accuracy = round(accuracy_score(y_test,y_pred),4)*100 
        accuracy = str(accuracy)+"%"
        report = classification_report(y_test, y_pred,  zero_division=1)
        cm=confusion_matrix(y_pred,y_test)
        f1 = round(f1_score(y_test,y_pred,average='macro'),3)
      
        acc=[]
        size=[]
        
        for k in range(2,10):
            k = k/10
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = k, random_state=42)
            X_train_text = X_train[text_columns].apply(lambda x: ' '.join(x.dropna()), axis=1)
            X_test_text = X_test[text_columns].apply(lambda x: ' '.join(x.dropna()), axis=1)
            X_train_vectorized = vectorizer.fit_transform(X_train_text)
            X_test_vectorized = vectorizer.transform(X_test_text)
            model.fit(X_train_vectorized, y_train)
            y_pred = model.predict(X_test_vectorized)
            size.append(k)
            acc.append(round(accuracy_score(y_test,y_pred),2))
            f1_score_svc.append(round(f1_score(y_test,y_pred,average='macro'),2))
            recall_score_svc.append(round(recall_score(y_test,y_pred,average='macro'),2))
            precision_score_svc.append(round(precision_score(y_test,y_pred,average='macro'),2))
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        pastel_blue_palette = sns.color_palette(["#D4FAFA", "#AFD5F0", "#AFD5F0"])
        sns.heatmap(cm, annot=True, fmt='d', cmap=pastel_blue_palette, cbar=False)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title('Confusion Matrix')
        cm = os.path.join(app.static_folder,'svc_cm.png')
        plt.show()
        plt.savefig(cm)
        plt.close()

        fig = plt.figure(figsize=(5.5, 4))
        fig.patch.set_facecolor('white') 
        plt.plot(size,acc, color='#AFD5F0')
        plt.xlabel("Training Size",color='black')
        plt.ylabel("Validation Accuracy",color='black')
        svc_f1_score = os.path.join(app.static_folder, 'svc.png')
        plt.show()
        plt.savefig(svc_f1_score)
        plt.close()

        image_files_to_delete = [cm, svc_f1_score]
        delete_image_files(image_files_to_delete, delay=7)
        
        return jsonify({'success': True, 'report': report, 'accuracy':accuracy,'stop':stop,'graph':'/static/svc.png','cm':'/static/svc_cm.png','f1':f1,'f1score':f1_score_svc, 'recallscore':recall_score_svc,'precisionscore':precision_score_svc})
    
    except Exception as e:
        return jsonify({'error': str(e)})

   

@app.route('/decision-tree', methods=['POST'])
def perform_decision_tree():
    global df_test
    global df_filtered_trimmed
    global f1_score_dt
    global recall_score_dt
    global precision_score_dt
    recall_score_dt.clear()
    precision_score_dt.clear()
    f1_score_dt.clear()

    try:
        start = time.time()
        X = df_filtered_trimmed.drop(columns=['Label'])
        y = df_filtered_trimmed['Label']
        X_text = X[text_columns].apply(lambda x: ' '.join(x.dropna()), axis=1)

        # Load the logistic regression model from pickle file
        with open('decision_tree.pkl', 'rb') as file:
            model = pickle.load(file)
        # Load the vectorizer from pickle file
        with open('vectorizer_fn.pkl', 'rb') as file:
            vectorizer = pickle.load(file)

        # Transform the test data using the vectorizer
        X_test_vectorized = vectorizer.transform(X_text)

        # Split the dataset into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_test_vectorized, y, train_size=0.8, random_state=42)
        y_pred = model.predict(X_test)

        stop = round(time.time() - start,4)
        accuracy = round(accuracy_score(y_test,y_pred),4)*100 
        accuracy = str(accuracy)+"%"
        report = classification_report(y_test, y_pred,  zero_division=1)
        cm=confusion_matrix(y_pred,y_test)
        f1 = round(f1_score(y_test,y_pred,average='macro'),3)
      
        acc=[]
        size=[]
        
        for k in range(2,10):
            k = k/10
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = k, random_state=42)
            X_train_text = X_train[text_columns].apply(lambda x: ' '.join(x.dropna()), axis=1)
            X_test_text = X_test[text_columns].apply(lambda x: ' '.join(x.dropna()), axis=1)
            X_train_vectorized = vectorizer.fit_transform(X_train_text)
            X_test_vectorized = vectorizer.transform(X_test_text)
            model.fit(X_train_vectorized, y_train)
            y_pred = model.predict(X_test_vectorized)
            size.append(k)
            acc.append(round(accuracy_score(y_test,y_pred),2))
            f1_score_dt.append(round(f1_score(y_test,y_pred,average='macro'),2))
            recall_score_dt.append(round(recall_score(y_test,y_pred,average='macro'),2))
            precision_score_dt.append(round(precision_score(y_test,y_pred,average='macro'),2))
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        pastel_blue_palette = sns.color_palette(["#D4FAFA", "#AFD5F0", "#AFD5F0"])
        sns.heatmap(cm, annot=True, fmt='d', cmap=pastel_blue_palette, cbar=False)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title('Confusion Matrix')
        cm = os.path.join(app.static_folder,'dt_cm.png')
        plt.show()
        plt.savefig(cm)
        plt.close()

        fig = plt.figure(figsize=(5.5, 4))
        fig.patch.set_facecolor('white') 
        plt.plot(size,acc, color='#AFD5F0')
        plt.xlabel("Training Size",color='black')
        plt.ylabel("Validation Accuracy",color='black')
        dt_f1_score = os.path.join(app.static_folder, 'dt.png')
        plt.show()
        plt.savefig(dt_f1_score)
        plt.close()

        image_files_to_delete = [cm, dt_f1_score]
        delete_image_files(image_files_to_delete, delay=7)
        
        return jsonify({'success': True, 'report': report, 'accuracy':accuracy,'stop':stop,'graph':'/static/dt.png','cm':'/static/dt_cm.png','f1':f1,'f1score':f1_score_dt, 'recallscore':recall_score_dt,'precisionscore':precision_score_dt})
    
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/f1score', methods=['POST'])
def f1score():
    global f1_score_lg
    global f1_score_nb
    global f1_score_rf
    global f1_score_svc
    global f1_score_dt

    global recall_score_lg
    global recall_score_nb
    global recall_score_rf
    global recall_score_svc
    global recall_score_dt

    global precision_score_lg
    global precision_score_nb
    global precision_score_rf
    global precision_score_svc
    global precision_score_dt

    size = [2,3,4,5,6,7,8,9]

    fig = plt.figure(figsize=(4.5, 4))
    fig.patch.set_facecolor('white') 
    plt.plot(size, f1_score_lg, color='#AFD5F0', label='Logistic Regression')
    plt.plot(size, f1_score_rf, color='#77dd77', label='Random Forest')
    plt.plot(size, f1_score_svc, color='#ff6961', label='SVC')
    plt.plot(size, f1_score_dt, color='#3D426B', label='Decision Tree')
    plt.xlabel("Relative Training Size",color='black')
    plt.ylabel("F1 Score",color='black')
    plt.legend()
    f1_score_all = os.path.join(app.static_folder, 'f1_score_all.png')
    plt.show()
    plt.savefig(f1_score_all)
  

    fig1 = plt.figure(figsize=(4.5, 4))
    fig1.patch.set_facecolor('white') 
    plt.plot(size, recall_score_lg, color='#AFD5F0', label='Logistic Regression')
    plt.plot(size, recall_score_rf, color='#77dd77', label='Random Forest')
    plt.plot(size, recall_score_svc, color='#ff6961', label='SVC')
    plt.plot(size, recall_score_dt, color='#3D426B', label='Decision Tree')
    plt.xlabel("Relative Training Size",color='black')
    plt.ylabel("Recall Score",color='black')
    plt.legend()
    recall_score_all = os.path.join(app.static_folder, 'recall_score_all.png')
    plt.show()
    plt.savefig(recall_score_all)

    fig2 = plt.figure(figsize=(4.5, 4))
    fig2.patch.set_facecolor('white') 
    plt.plot(size, precision_score_lg, color='#AFD5F0', label='Logistic Regression')
    plt.plot(size, precision_score_rf, color='#77dd77', label='Random Forest')
    plt.plot(size, precision_score_svc, color='#ff6961', label='SVC')
    plt.plot(size, precision_score_dt, color='#3D426B', label='Decision Tree')
    plt.xlabel("Relative Training Size",color='black')
    plt.ylabel("Precision Score",color='black')
    plt.legend()
    precision_score_all = os.path.join(app.static_folder, 'precision_score_all.png')
    plt.show()
    plt.savefig(precision_score_all)
    plt.close()

    image_files_to_delete = [f1_score_all, recall_score_all, precision_score_all]
    delete_image_files(image_files_to_delete, delay=7)
   

    return jsonify({'success':True, 
                    'graph': '/static/f1_score_all.png',
                    'graph1': '/static/recall_score_all.png',
                    'graph2': '/static/precision_score_all.png',
                    'f1_score_lg':recall_score_lg,
                    'f1_score_rf':recall_score_rf,
                    'f1_score_svc':recall_score_svc,
                    'f1_score_dt':recall_score_dt})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
