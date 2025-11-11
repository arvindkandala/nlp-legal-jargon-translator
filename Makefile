.PHONY: install train-simplifier train-classifier eval

install:
\tpython -m pip install -U pip
\tpip install -r requirements.txt

train-classifier:
\tpython src/train_classifier.py

train-simplifier:
\tpython src/train_simplifier.py

eval:
\tpython src/evaluate.py
