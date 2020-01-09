# Exercises, Data Mining at Aarhus University
The [Data Mining course](https://kursuskatalog.au.dk/da/course/95439/Data-Mining) 
at Aarhus University is primarily based on the 
[Zaki and Meira book](http://www.dataminingbook.info/pmwiki.php/Main/BookDownload):

> Zaki, M.J., Meira Jr, W. and Meira, W., 2014. 
> Data mining and analysis: fundamental concepts 
> and algorithms. Cambridge University Press.

Note that an online version of the book can be downloaded on the official
webpage, which is linked above. Furthermore, under the `Resources` tab on that
website, there are links to lecture videos, which might have value for you. 

> **Disclaimer:** the lecture videos are _not_ part of the course material and
> are not guaranteed to cover the same aspects of the course material as the
> actual lectures. So use them with cause.

Additional material for the course can be found on Black Board.

## Structure of the repository
Every week, there will be a Jupyter Notebook with exercises. These can be found
in the [exercises directory](./exercises). 

The [utilities directory](./utilities) includes the data that we will be
working with along with convenience methods for the data.

## Practical considerations 
### Tools
_note_: This course is a Python course, so in case you are not familiar with
Python, you might want to familiarize your self with Python and
JupyterNotebooks. Also, one library, that we are going to be using a lot is
`numpy`, which is a library that allows us to work with vectors and matrices.

### Setup
If you don't have Python installed already, we recommend you to install 
[MiniConda](https://docs.conda.io/en/latest/miniconda.html). MiniConda allows
you to have different environments (think different python installations) for
each of your projects, such that you can keep dependencies separated.

Install MiniConda and then open a conda terminal. In there, you can then create
an environment, where we will install the necessary packages for this course.

_Navigate to the project directory_:
```bash
> cd /path/to/dm20_exercises
```

_Create and activate environment:_  
```bash
> conda env create -f requirements.yml
> conda activate dm20
``` 

Now you should have created a conda environment with the necessary
dependencies.  From now on, when you want to run a python script or start a
notebook for this course, make sure to activate the environment (as in the last
line of coda above).  You know that your environment is active, if your active
line in the terminal is prefixed with `(dm20)`.

**Starting JupyterNotebook:**  
To start a jupyter notebook, navigate to the root of this repo and run the
following command from the command line:

```bash
(dm20) > jupyter notebook
```

The command should open a new window in your browser, where you can start running
Python scripts.

Happy hacking. 
