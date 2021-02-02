# Creating Virtual Environments with virtualenv

## Installation

`python3 -m pip install --upgrade pip`

`python3 -m pip install --user virtualenv` 

* Check which python3 is currently running:

`which python3`
- /usr/bin/python3

* create new environment specifying which python you want to use, so the one from the above line result. 

`virtualenv -p /usr/bin/python3 venv`

and then activate it with:

`source venv/bin/activate` 


## Freeze environment

`pip freeze > requirements.txt`


