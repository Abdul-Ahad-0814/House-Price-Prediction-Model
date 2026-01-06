from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("House_Prediction_Model.pkl", "rb"))

AREA_TO_SCORE = {
    # Premium
    "DHA": 5,
    "Clifton": 5,

    # Upper-middle
    "Gulshan-e-Iqbal": 4,
    "PECHS": 4,
    "North Nazimabad": 4,

    # Middle
    "Federal B Area": 3,
    "Johar": 3,

    # Lower-middle
    "Malir": 2,
    "Korangi": 2,

    # Low
    "Lyari": 1,
    "Orangi": 1
}



@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    beds = int(request.form["beds"])
    baths = int(request.form["baths"])
    area_sqyd = float(request.form["area"])
    area_name = request.form["area_name"]

    # Convert area name → location score
    location_score = AREA_TO_SCORE.get(area_name, 3)

    input_data = pd.DataFrame([{
        "Beds": beds,
        "Baths": baths,
        "Area": area_sqyd,
        "Location score": location_score
    }])

    prediction = model.predict(input_data)[0]

    return render_template(
        "index.html",
        prediction=round(prediction, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)
