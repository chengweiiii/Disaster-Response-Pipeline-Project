import pandas as pd
import json

#f = open("../models/gensim_summary.txt", "r")
#text = f.read().split("\n")
#json_string = json.dumps(text)
#print(json_string)
#f.close()
with open("../models/gensim_summary.txt") as f:
    text = f.read().split("\n")