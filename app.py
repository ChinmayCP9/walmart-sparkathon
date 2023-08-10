from flask import Flask, render_template
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    # Run the Python script and capture its output
    script_output = run_notebook_script()

    return render_template('index.html', script_output=script_output)

def run_notebook_script():
    # Run the converted Python script and capture its output
    result = subprocess.run(['python', 'salesvspositive.py'], stdout=subprocess.PIPE, text=True)
    script_output = result.stdout

    return script_output

if __name__ == '__main__':
    app.run(debug=True)