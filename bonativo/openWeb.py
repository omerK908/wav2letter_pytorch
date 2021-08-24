import os
from flask import Flask, render_template, request
import test
from pydub import AudioSegment as am

app = Flask(__name__, template_folder='template')

ALLOWED_EXTS = {"wav", "mp3"}


def check_file(file):
    return '.' in file and file.rsplit('.', 1)[1].lower() in ALLOWED_EXTS


@app.route("/", methods=["GET", "POST"])
def web():
    error = None
    filename = None
    predicted_text = None
    if 'POST' in request.method:
        if 'file' not in request.files:
            error = "file not selected"
            return render_template("index.html", error=error)

        file = request.files['file']
        filename = file.filename

        if ".mp3" in filename:
            filename = filename.split(".")[0]
            filename += ".wav"

        if filename == '':
            error = "File name is empty"
            return render_template("index.html", error=error)
        if not check_file(filename):
            error = "Upload only mp3 or wav files"
            return render_template("index.html", error=error)
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        # print(ROOT_DIR)
        file.save(os.path.join(ROOT_DIR + "/waves", filename))

        sound = am.from_file(ROOT_DIR + "/waves/" + filename, format='wav')  # change sample rate
        sound = sound.set_frame_rate(8000)
        sound.export(ROOT_DIR + "/waves/" + filename, format='wav')

        csvName = makeCsvForWav(ROOT_DIR + '/waves/' + filename)  # make csv file for the test
        predicted_text = test.testForWeb(csvName)
        print(predicted_text)
    return render_template("index.html", filename=filename, predict=predicted_text)


def makeCsvForWav(wavPath):
    csvName = 'test_csv.csv'
    file1 = open('csvFile/' + csvName, "w")
    file1.write(wavPath + ',a')
    file1.close()
    return csvName

if __name__ == "__main__":
    app.run()
