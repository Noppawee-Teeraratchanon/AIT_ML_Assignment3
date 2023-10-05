from dash import Dash, dcc, html, Input, Output, State, callback
import pickle
import warnings
import numpy as np
import pandas as pd
import pickle
warnings.filterwarnings('ignore')
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


#experiment tracking
import mlflow
import os
# This the dockerized method.
# We build two docker containers, one for python/jupyter and another for mlflow.
# The url `mlflow` is resolved into another container within the same composer.
mlflow.set_tracking_uri("https://mlflow.cs.ait.ac.th/")
mlflow.set_experiment(experiment_name="st124482-a3")



# load the model a3
model = mlflow.pyfunc.load_model('runs:/1266fcd065a34fb8a3598bfb8c1d9ded/model/')

# make a list of transmission
list_transmission = ["Automatic", "Manual", "None"]



app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1("Predicting Car Price System", style={'textAlign': 'center'}),
    html.Br(),
    html.H6("Instruction", style={'color':'Blue'}),
    html.H6("There is classification model on this system. the model need only 3 values to predict the selling car price class including:", style={'color':'Blue', 'margin-left': '50px'}),
    html.H6("1. Max power: Users assign the max power of the car. If you are not sure about max power of the car, you can leave it blank.", style={'color':'Blue', 'margin-left': '50px'}),
    html.H6("2. Mileage: Users assign the mileage of the car. If you are not sure about the mileage of the car, you can leave it blank.", style={'color':'Blue', 'margin-left': '50px'}),
    html.H6("3. Transmission: Users assign the type of transmission of the car. If you are not sure about the transmission type of the car, you can leave it blank or choose None.", style={'color':'Blue', 'margin-left': '50px'}),
    html.H6("After you fill in all of information that you know, you click the submit button, and the website will show class of the selling price of this car at the bottom of the page.", style={'color':'Blue', 'margin-left': '50px'}),
    html.H6("The lowest to highest price of car: class 0 < class 1 < class 2 < class 3 ", style={'color':'Blue', 'margin-left': '50px'}),
    html.H6("Max Power : ",  style={'display':'inline-block', 'margin-left': '300px'}),
    dcc.Input(id='input-max_power-state', type='number'),
    html.Br(),
    html.H6("Mileage : ",  style={'display':'inline-block', 'margin-left': '300px'}),
    dcc.Input(id='input-mileage-state', type='number'),
    html.Br(),
    html.H6("transmission : ",  style={'display':'inline-block', 'margin-left': '300px'}),
    dcc.Dropdown(id='input-transmission-state', options=[{'label': list_transmission[i], 'value': i} for i in range(len(list_transmission))], style={'display':'inline-block','width':'50%'}),
    html.Br(),
    html.Br(),
    html.Button(id='submit-button-state', n_clicks=0, children='Submit', style={'width': '200px', 'margin-left': '625px', 'color': 'Red', 'background': 'White'}),
    html.Br(),
    html.Br(),
    html.Br(),
    html.H4(id='output-a3', style={'color':'Blue', 'margin-left': '50px'}),
    ], style = { 'background': 'Silver'})



@callback(Output('output-a3', 'children'),
              Input('submit-button-state', 'n_clicks'),
              State('input-max_power-state', 'value'),
              State('input-mileage-state', 'value'),
              State('input-transmission-state', 'value'),
)
              
# create output function to predict the selling price of a car
def update_output(n_clicks,max_power,mileage,transmission):  
    
    if n_clicks>=1:
        if max_power == None:
            max_power = 82.85 #default of max power is median of training data which is 82.85
        if mileage == None:
            mileage = 19.38  #default of mileage is mean of training data which is 19.38
        if transmission == None or transmission == 2:
            transmission = 1    #default of transmission is mode of training data which is 1 (Manual)
        
        
        scaler = pickle.load(open('/root/code/scaler.pkl', 'rb'))
        [max_power, mileage]  = scaler.transform([[max_power,mileage]])[0]
        sample_a3 = np.array([[max_power, mileage, transmission]])
        intercept = np.ones((1, 1))
        sample_a3   = np.concatenate((intercept, sample_a3), axis=1)
        
        return ('the car is on class ', int(model.predict(sample_a3)))
if __name__ == '__main__':
    app.run(debug=True)