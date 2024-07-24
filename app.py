import flask
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the model

import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv('USA_Housing.csv')
X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']]
y = df['Price']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the model
with open('model/USA_Housing_Model.pkl', 'wb') as f:
    pickle.dump(model, f)


app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return flask.render_template('index.html')
    
    if flask.request.method == 'POST':
        aai = float(flask.request.form['aai'])
        aah = float(flask.request.form['aah'])
        aan = float(flask.request.form['aan'])
        aanb = float(flask.request.form['aanb'])
        ap = float(flask.request.form['ap'])
        
        input_variables = pd.DataFrame([[aai, aah, aan, aanb, ap]],
                                       columns=['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population'],
                                       dtype=float,
                                       index=['input'])
        
        prediction = model.predict(input_variables)[0]
        
        return flask.render_template('index.html',
                                     original_input={'Avg. Area Income': aai,
                                                     'Avg. Area House Age': aah,
                                                     'Avg. Area Number of Rooms': aan,
                                                     'Avg. Area Number of Bedrooms': aanb,
                                                     'Area Population': ap},
                                     result=prediction)

if __name__ == '__main__':
    app.run(debug=True)
