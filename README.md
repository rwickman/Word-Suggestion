# Word-Suggestion
## Overview
A language model built with an RNN that will predict the next word given a sequence of words.

There is an option to  find similar users based upon what you are currently typing and use their word suggestion model to help make predictions. It will use LSH with containment similarity, the algorithm I came up with is described in the report docs/Report/Word_Suggestions_Report.pdf under the Algorithm section.

This project was done for COMP 4118 Data Mining.

## Installation
Python 3.x and Pip is required to run/install this project which can be found [here](https://www.python.org/)

Furthermore, you need to the run the following command to install the dependencies:
```
pip install datasketch, tensorflow, websockets, numpy 
```

## How to Run
Under the python directory there is a main.py file. If you only want to use the single word suggestion model, first run:
```
python main.py
```

If instead you want to include the LSH suggestions, run:
```
python main.py --find-simlar
```

After a few seconds the following phrase should appear:
> READY TO SERVE PREDICTIONS. YOU MAY NOW START UP THE WEBPAGE.

Now start up webpage/index.html in your choice of web browser.

If everything was installed properly, it should present you with up to 5 suggested word. These are ranked most to least likely, from left to right respectively.

While you start typing, or select a word from the list, it should give you more suggestions based on what you entered. 
*There may be a small delay depending on the computer you are using*

## Algorithm
This can be found in the report docs/Report/Word_Suggestions_Report.pdf under the Algorithm section.