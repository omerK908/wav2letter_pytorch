import os
from flask import Flask, render_template, request
import test

app = Flask(__name__, template_folder='template')

ALLOWED_EXTS = {"wav", "mp3"}


def check_file(file):
    return '.' in file and file.rsplit('.', 1)[1].lower() in ALLOWED_EXTS


@app.route("/", methods=["GET", "POST"])
def web():
    error = None
    filename = None
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
        print(ROOT_DIR)
        file.save(os.path.join(ROOT_DIR + "/waves", filename))
        # TODO run the test on the file

        csvName = makeCsvForWav(ROOT_DIR + '/waves/' + filename)
        print(csvName)
        print(test.testForWeb(csvName))
    return render_template("index.html", filename=filename)


def makeCsvForWav(wavPath):
    import csv
    csvName = 'test_csv.csv'

    # open the file in the write mode
    # f = open('csvFile/' + csvName, 'w')
    #
    # # create the csv writer
    # writer = csv.writer(f)
    #
    # # write a row to the csv file
    # writer.writerow(wavPath)
    # f.close()

    file1 = open('csvFile/' + csvName, "w")

    # \n is placed to indicate EOL (End of Line)
    file1.write(wavPath)
    file1.close()  # to change file access modes



    return csvName

if __name__ == "__main__":
    app.run()
