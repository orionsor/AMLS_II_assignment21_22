# AMLS_II_assignment21_22_21051323

The aim is the assignment is to solve subtask A and C in SemEval 2017 Task 4. An LSTM-Bi-LSTM model is proposed for Task A "Message Polarity Classification", and an multi-input Bi-LSTM model is built for Task C "Top-based Polarity Classification on 5-point Scale". Original data underwent a series of preprocessing, including data cleansing, tokenization, stemming, lemmatization, emoticon recognition,etc. Both model obtained proper result, with an accuracy of 63.58% and 64.68% for Best model in two subtasks respectively, which would rank high in comparison with solution in real competition. Model complexity is balanced with performance.

## File Organization
 "main.py" in the root branch is for running the main content of the assignment, which includes data preprocesssing, model training, validating and testing of the optimal models finetuned with validation set for the corresponding task. File folder "Dataset" contains original datasets and model or plots newly generated. Necessary 
 file for each task is save in the corresponding file folder "A" and "C".
  
 ## Files In ./Datasets
 Files in this folder are involved with binary classification task, which can be devided into 3 part in terms of role. 
 ### ./Datasets/w2v_model
 GloVe word to vector model is saved for building word embedding layer and vectorize data
 ### ./Datasets/A
 
 1.'./arabic/SemEval2017-task4-train.subtask-A.arabic.txt' is the Arabic dataset used in Task A to evaluate naive bayes model.
 2. './english/twitter-2016train-A.txt' and './english/twitter-2016train-A.txt' is the English dataset in Task A, which will be concatenated as the raw dataset.
 3. './english/model/' will save model for task A when running 'main.py'
 ### ./Datasets/C
 
 1. likewise, "twitter-2016train-CE.txt" and "twitter-2016train-CE.txt" are the dataset for task C
 2. ./model/ is to save model for Task C when running 'main.py'
 
  ## Files In ./A
 Though only the best model is implemented in "main.py", other model for experiments is included in the branch and can be run seperately.
 1. 'data_process_ekphrasis.py' and 'BiLSTM.py' is wrote for the running of 'mainly', which includes data preprocessing with special character recognition and the implementation of the best model. These two file can run without 'main.py': run data_process file first to generare processed data, then run the model file for training and testing.
 2. run 'data_process_en_aug.py' and'Bilstm_token_aug.py' in order, performance on data preprocessung with data augmentation and without extra recognition can be viewed.
 3. run 'data_process_no_aug.py' and'Bilstm_token.py' in order, performance on data with basic data preprocessing can be viewed.
 4. 'en_baseline_naive_bayes.py' is the implementation of multinominal naive bayes on english dataset.
 5. 'arab_naive_bayes.py' is the implementation of multinominal naive bayes on arabic dataset.
 6. ./plot/ saves graphs such as word cloud generated in experiments, learning curves are also saved here for check.
 7.  ./model/ will save the newly-optimized models if the above files are implemented.

 ##  Files In ./C
 the organization is basically the same as ./A
 1."c_data_process.py" and "c_biLSTM_ek.py" are the implementation of optimal preprocessed data and model to be run on 'main.py'.
 2. "c_data_process_basic.py" and "c_biLSTM_basic.py" is the comparison model where data is not processed with ekphrasis.

 
 ## Necessary Library
 the code run locally with python 3.8. following libriaries are used:
 keras==2.6.0, tensorflow==2.6.0, os, numpy, pandas, matlotlib, nltk, pickle, ekphrasis, gensim==4.1.2, re, string, sklearn, time
