import contextlib
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from msg_parser import MsOxMessage
from bs4 import BeautifulSoup
from datetime import datetime
import lxml

def clear_and_create(*paths: Path):
    for path in paths:
        if path.exists():
            shutil.rmtree(str(path))
        path.mkdir()


data = Path("E:/docs/yo_la_tengo")

parsed = Path("data/parsed")
raw = Path("data/raw")
clear_and_create(parsed, raw)


mails = []
GPT_DELIMITER = "<|endoftext|>"

# BODY_PATTERN = re.compile("Hello all, (.*?)Allons-y!")

@dataclass
class YLTMail:
    sent: datetime
    body: str
    parsed_body: str


def remove_bad_chars(r: str) -> str:
    return


def cutout_body(p: str) -> str:
    try:
        start = p.index(",") + 1
        end = p.index("-y!")
        return p[start:end].strip()
    except:
        raise AttributeError

for f in data.iterdir():
    with contextlib.suppress(AttributeError):
        msg = MsOxMessage(f)
        # msg.sent_date
        # Sun, 4 Apr 2021 15:24:06 +0000

        body = BeautifulSoup(msg.body, "lxml").text
        body = remove_bad_chars(body)
        body = cutout_body(body)
        # body = body.strip()

        mails.append(YLTMail(
            sent=datetime.strptime(msg.sent_date, "%a, %d %b %Y %H:%M:%S %z"),
            body=msg.body,
            parsed_body=body
        ))

mails.sort(key=lambda m: m.sent)
all = ""
for mail in mails:
    filename = parsed / (mail.sent.strftime("%Y-%m-%d") + ".txt")

    text = mail.parsed_body + GPT_DELIMITER

    all += text

with (parsed / "all.txt").open("w") as f:
    f.write(all)
