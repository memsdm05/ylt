import multiprocessing
import threading
import queue
import random

from flask import Flask, url_for, render_template
import argparse
import generator

parser = argparse.ArgumentParser(
    description="Arguments for the running the server independently"
)
parser.add_argument("-p", "--port", default=8080)
parser.add_argument("-a", "--host", default="127.0.0.1")
parser.add_argument("--tokenizer")
parser.add_argument("--model")
args = parser.parse_args()

app = Flask(__name__)


class ParallelGenerator:
    starters = [
        "",
        "",
        "",
        "I hope",
        "Welcome to",
        "The",
        "This is",
        "I",
        "The pandemic",
    ]

    def __init__(self, buffer=20):
        self.gen = generator.YLTGenerator(
            tokenizer_path=args.tokenizer,
            model_path=args.model
        )

        self.min = 100
        self.max = 300

        self.size = 0
        self.texts = queue.Queue()
        # self.texts = multiprocessing.Queue(maxsize=buffer)

    def start(self, spawn=2):
        for _ in range(spawn):
            threading.Thread(target=self._runner).start()
            # multiprocessing.Process(target=self._runner).start()

    def _runner(self):
        while True:
            self.texts.put(
                self.gen(random.choice(self.starters))
            )
            self.size += 1
            print(self.size)
        # while True:
        #     text = self.gen(random.choice(self.starters), max_length=self.max, min_length=self.min)
        #     self.texts.put(text)


pg = ParallelGenerator()

@app.route("/", methods=["GET"])
def index():
    if pg.size == 0:
        return "wait please", 502
    pg.size -= 1
    print("OUT", pg.size)

    text = pg.texts.get()
    if not text.endswith("."):
        text += "."

    if not text.endswith("Be Kind!"):
        text += " As always, Laugh Hard... Run Fast... Be Kind!"

    return "Good Sunday morning all, <br><br>" + text + "<br><br>Allons-y!"


if __name__ == "__main__":
    pg.start(spawn=1)
    print("here")
    app.run(debug=False, host=args.host, port=args.port)
