# Word-Suggestion
## Overview
A language model that will predict the next word given a sequence of words.  

There is an option to find similar users and use their language model to make predictions based upon what you are currently typing. This is done by utilizing LSH with containment similarity, the algorithm I came up with is described in detail below.

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

If everything was installed properly, it should present you with up to 5 suggested word. These are ranked from most to least likely, left to right respectively.

While you start typing, or select a word from the list, it will update the suggestions list with new suggestions based on what was entered. 
*There may be a small delay depending on the computer you are using.*

## Algorithm
User = U, Similar User (or a subset of the Similar User) = P

Step 1 (Find Similar Users):

&nbsp;Each time U enters in a word:

&nbsp;&nbsp;&nbsp;Compute the containment similarity between the content of the text box (i.e., what the user as entered so far) and all P.

&nbsp;If the similarity exceeds a given threshold:

&nbsp;&nbsp;&nbsp;Use P’s RNN to make word suggestions

&nbsp;&nbsp;*NOTE: In some cases Multiple P’s will share the same RNN, in these cases only need to make the prediction once*

Step 2 (Update):

&nbsp;If U uses a word suggested from P’s RNN:

&nbsp;&nbsp;&nbsp;Update U’s RNN by training on the subset of data associated with P 

&nbsp;&nbsp;&nbsp;Update U’s RNN with what the sequence of words the user as entered

> **EX:** P = song lyrics</br>
U = user</br>
U picks a word suggested from an RNN trained on P</br>
Then train on the subset of songs from P with similarity that exceeds the similarity threshold</br>
