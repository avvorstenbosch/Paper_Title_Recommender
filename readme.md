# Paper Title Recommender
*This repo is a work in progress, and currently not yet usable as described below.*
This repo contains the code to train and apply a paper title recommender system.
There are two title generation modes:
* __MLE title__\
In this mode, the most probable (maximum likelihood estimator) title is selected via a (greedy) beamsearch algorithm.
* __Random title__\
In this mode, the user can specify the Temperature and generate titles based on random sampling



# How to use

There are two main ways to run the code.
The first mode is by specifying the required text fields via de CLI:\
```$python main --category="CATGEGORY" --abstract="ABSTRACT" --name="FILENAME"```\
\
A more convenient, which supports batch processing, is refering to a .csv file:\
```$python main --file="./inputs/MYFILES.csv" --name="FILENAME"```\
\
The results are saved at `./input/FILENAME.txt` and `./input/FILENAME.json`\
\
An example of the generated output is shown here:\
`TODO output example`\
`GPT2-medium` succeeds at producing good sounding titles based on the given context.
One should note that many of the titles produces clearly don't understand the science being discussed.
So while the titles generated sound like plausible titles, and may in fact be very suitable, many of the generated titles don't make scientific sense.
From manual inspection it appears that the quality of titles deteriorates with shorter abstracts. This makes sense as the algorithm has less context to work with.
In these cases it is also the case that a non-expert could not have been generated the actual title given the context.\
\
I recommend you use this algoritm as a way to boost your own creativity and hopefully be inspired by some of the suggestions. \

# How it works

This project makes use of the `Huggingface` library to perform abstractive title generation.\
This is achieved by finetuning `GPT2` on the `Arxiv Metdata dataset`.\
For my own experiments, I finetuned the network for roughly 20.000 timesteps with a batchsize of 8.\
Different values might be more optimal, but the results generated with these settings were pleasing upon manual inspection of random samples.




