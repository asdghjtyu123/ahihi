import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.datasets import Multi30k
from torchtext.data import Field, TabularDataset

import spacy
import numpy as np

import random
import math
import time

spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')
def tokenize_de(text):
    
    return [tok.text for tok in spacy_de.tokenizer(text)]
