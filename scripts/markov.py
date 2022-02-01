import random
import string

ngrams = {}
valid_chars = string.ascii_lowercase + string.ascii_uppercase + string.digits + "',!?. []\n"
NGRAM_SIZE = 5
MAX_INPUT = 0

def build():
    r = 0
    c = ""
    ngram = ""
    with open("../data/parsed/all.txt", "r") as f:
        while True:
            try:
                c = f.read(1)
            except:
                pass

            if not c or (MAX_INPUT > 0 and r > MAX_INPUT):
                break

            if c not in valid_chars:
                continue

            r += 1

            if len(ngram) < NGRAM_SIZE:
                ngram += c
            else:
                if ngram not in ngrams:
                    ngrams[ngram] = {}

                node = ngrams[ngram]
                if c not in node:
                    node[c] = 0

                node[c] += 1

                ngram = (ngram + c)[1:]


GEN_SIZE = 1000
def generate():
    ngram = "Hello"
    out = ngram
    for _ in range(GEN_SIZE):
        chars, weights = zip(*ngrams[ngram].items())
        next = random.choices(chars, weights)[0]

        out += next
        ngram = (ngram + next)[1:]

    print(out)



if __name__ == '__main__':
    build()
    generate()