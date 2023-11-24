from flask import Flask, render_template, request, jsonify
from app import app
from app.classificacao import LogisticRegression

model = LogisticRegression()

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if request.method == 'POST':
        # Obtendo os dados do formulário HTML
        features = [float(request.form[f'feature{i}']) for i in range(1, 31)]

        # Realizando a previsão usando o modelo treinado
        prediction = model.predict([features])

        # Convertendo o resultado para uma string legível
        result = "Maligno" if prediction[0] == 1 else "Benigno"

        # Criando um dicionário para retornar os resultados como JSON
        result_dict = {
            'result': result,
            'accuracy': float(model.accuracy),
            'conf_matrix': model.conf_matrix.tolist(),
            'class_report': model.class_report
        }

        return jsonify(result_dict)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
