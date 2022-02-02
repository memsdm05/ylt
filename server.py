from flask import Flask, request
import argparse
import sys

parser = argparse.ArgumentParser(
    description="Arguments for the running the server independently"
)
parser.add_argument("-p", "--port", action="store_const", const=8080)
parser.add_argument("-a", "--address", action="store_const", const="127.0.0.1")
parser.parse_args(sys.argv)

app = Flask(__name__)


# model = generator.load(tokenizer="", model="")


@app.route("/generate", methods=["GET"])
def api():
    start = request.args.get("start")
    return {"hello": start}


@app.route("/", methods=["GET"])
def index():
    return "Hello World"


if __name__ == "__main__":
    app.run(debug=True)
