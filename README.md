# Our Codes for the RecSys 2018 Challenge (6th Position on the Final Leaderboard)

 

This repository contains the **Python** source code of our solutions to the RecSys 2018 challenge.



## Data Preparation

In order to replicate our submissions to the RecSys 2018 Challenge, you first need to download the **Million Playlist Dataset** and the **Challenge Set** from the [official RecSys challenge website](https://recsys-challenge.spotify.com/). These two datasets are recorded as JSON files and compressed as `mpd.v1.tgz` and `challenge.v1.tgz`, respectively.  After downloading these files, please decompress them, and put the decompressed files in a new folder called `src/data/` . If done correctly, the JSON files of the **Million Playlist Dataset** should be then be stored in the sub-folder `src/data/mpd.v1/data/` and the JSON files of the **Challenge Set** should be stored in the sub-folder `src/data/challenge.v1/` .



## Code Structure

1. **Data preprocessing.** The large number of JSON files are a bit cumbersome for downstream data manipulation, thus we first processed them into a few CSV files, the used scripts are stored in the `src/utilities/` folder.
2. **Results generation for 10 subtasks.** This year's RecSys Challenge have 10 subtasks that are supplied with different amounts of observed data and meta information. For these tasks, we thus accordingly used predictive strategies that were somewhat different. The used codes were stored into 10 different folders, such as `src/5songs_with_title/`, `src/25songs_shuffle/`, and `src/10songs_with_title/`, etc. Their bijective correspondences with the 10 subtasks can be easily identified purely based on their names.  
3. **Summarization of the results into the submission file.** The used scripts are stored in the `src/submit/` folder. 



## Requirements

### Running Environment and Required Computational Resources.

Basically, we run our project under Python 3.6, on a Xeon E5-2683*2+320GB machine. 

The preprocessing part will take almost 4 hours and 100GB memory. Each subtask result will normally take about 2 hours and less than 180GB memory. However, for 1song, 5songs-with-title, 10songs-with-title, 10songs-without-title, we use collaborative filtering (CF) methods first, and then use these CF scores as one feature feed to Lightgbm to do the final prediction. The collaborative filtering method is therefore time-consuming,  since it will not only get CF-scores for 1000 playlists in the test set but also 12000 playlists in the train set for LGB fitting.  Resultantly, each of the 4 aforementioned sub-task will take almost **40 hours**, please keep patient.

### Dependencies

- Numpy/Scipy/Scikit-learn/Pandas
- Json (tested on v2.0.9) 
- Tqdm (tested on v4.23.4)
- Lightgbm (tested on v2.0.10)



## Results Generation 

To generate the results for our final solutions, you just need to run `python generate_results.py`, 
it will automatically execute the entire predictive process, which includes data preprocessing, results generation for 10 subtasks, and combination of the overall results as the required format of the challenge.

To get the whole result will take a very long time, just for convenience, if you are interested in generating results for a single subtask only, please note that each of the 10 subtask folders contains a python script   whose file name starts with 'auto', such as the `auto_10songs_with_title.py` file in the `src/10songs_with_title/` folder. You can just run such files to generate results for individual tasks independently. However, the `auto_generate.py`  in the `src/utilities/` folder still needs to be run beforehand to perform the preprocessing step. 



## License


Copyright [2018][Bowen He, Lin Zhu, Mengxin Ji, Cheng Ju]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.



## Contact


Our scripts are a little messy and we think there must be some potential improvements for our codes.
Please feel free to contact us, if there are any question about our project or suggestion to help improve.
email: hebowen_1994@hotmail.com, mji@ucdavis.edu
