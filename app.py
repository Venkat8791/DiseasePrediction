import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
from scipy.stats import mode

from flask import Flask, render_template, request

app = Flask(__name__)

#step-0
#load pkls
final_rf_model = pickle.load(open('rf.pkl', 'rb'))
final_nb_model = pickle.load(open('naive.pkl', 'rb'))
final_svm_model = pickle.load(open('svm.pkl', 'rb'))

#step-1
DATA_PATH = "Training.csv"
DESCRIPTION="Description.csv"
PRECAUTION = "Precaution.csv"
data = pd.read_csv(DATA_PATH).dropna(axis = 1)

desc = pd.read_csv(DESCRIPTION)
prec = pd.read_csv(PRECAUTION)

#step-2
# Encoding the target value into numerical
# value using LabelEncoder
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])

#step-3
X = data.iloc[:,:-1]
y = data.iloc[:, -1]
symptoms = X.columns.values

# Creating a symptom index dictionary to encode the
# input symptoms into numerical form
symptom_index = {}
for index, value in enumerate(symptoms):
	symptom = " ".join([i.capitalize() for i in value.split("_")])
	symptom_index[symptom] = index

data_dict = {
	"symptom_index":symptom_index,
	"predictions_classes":encoder.classes_
}

@app.route('/')
@app.route('/index',methods=['GET','POST'])

def index():
    return render_template('index.html')


@app.route('/predictDisease',methods=['POST'])
def predictDisease():
    if request.method=="POST":
        s1=request.form["symptom1"]
        s2=request.form["symptom2"]
        s3=request.form["symptom3"]
        s4=request.form["symptom4"]
        s5=request.form["symptom5"]
        print(s1)
        print(s2)
        print(s3)
        print(s4)
        print(s5)
        if s4=='':
            symptoms = [s1,s2,s3]
        elif s5=='':
            symptoms = [s1,s2,s3,s4]
        else:
            symptoms = [s1,s2,s3,s4,s5]
            # creating input data for the models
        input_data = [0] * len(data_dict["symptom_index"])
        for symptom in symptoms:
            index = data_dict["symptom_index"][symptom]
            input_data[index] = 1
                
            # reshaping the input data and converting it
            # into suitable format for model predictions
        input_data = np.array(input_data).reshape(1,-1)
            
            # generating individual outputs
        rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
        nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
        svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]
            
            # making final prediction by taking mode of all predictions
        final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])[0][0]
        print(rf_prediction)
        print(nb_prediction)
        print(svm_prediction)
        print(final_prediction)
        predictions = {
                "rf_model_prediction": rf_prediction,
                "naive_bayes_prediction": nb_prediction,
                "svm_model_prediction": svm_prediction,
                "final_prediction":final_prediction
        }
        description=desc.loc[desc['Disease'] == final_prediction, 'Description'].iloc[0]
        p1=prec.loc[prec['Disease'] == final_prediction, 'Precaution_1'].iloc[0]
        p2=prec.loc[prec['Disease'] == final_prediction, 'Precaution_2'].iloc[0]
        p3=prec.loc[prec['Disease'] == final_prediction, 'Precaution_3'].iloc[0]
        p4=prec.loc[prec['Disease'] == final_prediction, 'Precaution_4'].iloc[0]
        precaution = "1." + p1 + "\n" + "2." + p2 + "\n" + "3." + p3 + "\n" + "4." + p4 + "\n"
        return render_template('index.html',rf_model_prediction=rf_prediction,nb_model_prediction=nb_prediction,svm_model_prediction=svm_prediction,symptom1=s1,symptom2=s2,symptom3=s3,symptom4=s4,symptom5=s5,description=description,precaution=precaution)
    return render_template('index.html')

if __name__=='__main__':
    app.run(debug=True)
