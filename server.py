import multiprocessing
import queue
import random

from flask import Flask, url_for, render_template
import argparse
import sys

import generator

parser = argparse.ArgumentParser(
    description="Arguments for the running the server independently"
)
parser.add_argument("-p", "--port", default=8080)
parser.add_argument("-a", "--host", default="127.0.0.1")
parser.add_argument("--tokenizer")
parser.add_argument("--model")
args = parser.parse_args(sys.argv)

app = Flask(__name__)


class ParallelGenerator:
    starters = [
        "",
        "I hope",
    ]
    def __init__(self, spawn=3):
        self.gen = generator.YLTGenerator(
            tokenizer_path=args.tokenizer,
            model_path=args.model
        )

        self.texts = queue.Queue()

        for _ in range(spawn):
            multiprocessing.Process(target=self._runner)

    def _runner(self):
        while True:
            self.gen()



@app.route("/generate", methods=["GET"])
def generate():
    return {
        "text": "todo"
    }


@app.route("/", methods=["GET"])
def index():
    return render_template("./public/index.html")


if __name__ == "__main__":
    app.run(debug=True, host=args.host, port=args.port)
