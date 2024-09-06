from flask import Flask, render_template, request
import entradas as ent  # Seu módulo com a função de conversão

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error_message = None
    if request.method == 'POST':
        try:
            celsius = float(request.form['celsius'])
            # Formatar o resultado para 3 casas decimais
            result = round(ent.converter(celsius), 3)
        except ValueError:
            error_message = "Por favor, insira um número válido."

    return render_template('index.html', result=result, error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)
