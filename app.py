from flask import Flask, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

@app.route('/predice', methods=['POST'])
def predict():
    json_ = request.json

    # convierte a DataFrame
    query_df = pd.DataFrame([json_])

    # asegúrate de usar las mismas columnas que en el entrenamiento
    query_df = query_df[["age", "fare", "pclass", "sex", "embarked_C", "embarked_Q", "embarked_S"]]

    # carga el modelo
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'classifier.pkl')
    classifier = joblib.load(model_path)

    # predice
    pred = classifier.predict(query_df)

    if pred[0]:
        return "TRUE: El pasajero pudo haber sobrevivido"
    else:
        return "FALSE: El pasajero pudo NO haber sobrevivido"

if __name__ == "__main__":
    app.run(port=8000, debug=True)
