# Recsys-Spotify-2018-challenge
Source code of  our solutions to recsys-spotify 2018 challenge


Copyright [2018] [Bowen He, Lin Zhu, Mengxin Ji, Cheng Ju]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.



Running enviroment and memory required:
Basically, we run our project under python3.5, 
On Xeon E5-2683*2+320g machine,  the preprocessing part will take almost 3 hours and 100g memory.
Each subtask result will take almost 2 hours and 150g memory.
Other package required you can find in requirements.txt.


How to run the whole source code?
To generate the final results for our final solutions, you just need to run "python generate_results.py", 
it will automatically generate results for 10 subtasks, and combine it as required format in the challenge.

To get the whole result will take a very long time, just for convenience, if readers are interested in how we get
each result, you can just run the auto-prefix file in each sub-directroy, you can tell which sub task in which directory
just by the name. 
BUT MAKE SURE YOU HAVE RUN THE AUTO-PREFIX FILE BEFOREHAND!


What algorithm we taks for each sub-task?
The main method behind our solution is collaborative filtering, more precise, we took item-based collaborative filtering.
I will give a brief idea for our solutions for each sub-task.

For the 'title-only' predict task:
First we do some nlp tricks on the titles, and we just treat the title and corresponding tracks as words(make up a title)
-tracks co-matrix, and use this co-occurence relation to predict, just a novel variation of item-based cf.

For the '1song', '5songs'(two, title and without title), '10songs'(two, title and without title) sub-tasks;
We use the item-based cf, to get first 1000 top tracks, then used lightgbm to rank the 1000 again, to get
the most possible 500 songs.
(We do not use lgb for '5songs-without title' sub-task, since the further work not help so much compared with cf)

For the '25songs-in order' and '100songs-in order' prediction sub-tasks:
We add some special tricks related with sequences on item-based collaborative filtering.

For the '25songs shuffle' and '100songs shuffle' prediction sub-tasks:
We combine slim ideas with item-based collaborative filtering.


Our scripts is a little messy and we think there are must be some improvements among our codes.
Please feel free to contact us, if there are any question about our project or suggestion to help improve.
email: hebowen_1994@hotmail.com
