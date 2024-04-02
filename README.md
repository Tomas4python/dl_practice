# Deep Learning Framework Comparison: Practical Project 5

## Project Overview

The objective of this project is to compare the efficacy and features of various deep learning frameworks, specifically Keras, TensorFlow, PyTorch, and Fast.ai. 

## Datasets

The search for clean datasets led me to Kaggle, with the intention of bypassing extensive data preparation stages. Initially, a dataset for a classification task to diagnose diabetes was chosen. Despite favorable reviews, the dataset proved to be ineffectively cleaned, consisting mainly of common features with weak relevance to diabetes diagnosis. Consequently, achieving high accuracy was unattainable, leading to the abandonment of this dataset.

Subsequently, a smoke detection dataset was selected. However, this too was disappointing as all but one feature were found to be redundant. The sole significant feature enabled swift achievement of high accuracy, leaving little room for further exploration.

Returning to a familiar territory, I opted for a dataset from a my prior machine learning project—the green taxi dataset. This dataset, already cleaned and prepared for machine learning tasks, facilitated a linear regression task to predict total taxi fare amounts. This allowed for a direct comparison between machine learning and deep learning outcomes.

Links to the datasets are included within the respective notebooks.

## Project Components

- `diabetes_dl_keras.ipynb`: A Keras-based notebook for the classification task on the diabetes dataset.
- `smoke_dl_keras.ipynb`: A Keras-based notebook for the classification task on the smoke detection dataset.
- `green_taxi_dl_keras.ipynb`: A Keras-based notebook for taxi fare prediction.
- `green_taxi_dl_pytorch.ipynb`: A PyTorch-based notebook for taxi fare prediction.
- `green_taxi_dl_fastai.ipynb`: A Fast.ai-based notebook for taxi fare prediction.
- `model_performance_metrics_diabetes.xlsx`: An Excel file documenting the performance metrics of the diabetes model.
- `model_performance_metrics_green_taxi.xlsx`: An Excel file documenting the performance metrics of the green taxi models.

## Data Exploration and Preprocessing

The datasets chosen were pre-cleaned to expedite the project. The green taxi dataset was divided into training, validation, and test sets, similar to the splits used in my previous machine learning projects, to enable direct result comparisons.

## Framework Reflections and Conclusions

In Practical Project 5, the task at hand was to reflect upon and articulate a preference for one of the deep learning frameworks explored: Keras, TensorFlow, PyTorch, and Fast.ai. The task, however, is challenging as my experience does not incline towards clear likes or dislikes. Each framework is designed with specific purposes in mind and, in their own capacity, all have served me well. Keras and Fast.ai, being higher-level APIs, offered simplicity and a more user-friendly interface, which was beneficial for straightforward tasks.

Both PyTorch and TensorFlow displayed robust capabilities, and familiarizing oneself with either could be advantageous due to their extensive use in the industry. They each have comprehensive documentation and an array of courses available, which supports a deep dive into their functionalities.

Fast.ai, while equally powerful, seemed to have a narrower range of learning resources. I found it challenging to locate a resource that provided a succinct yet comprehensive overview of the framework. Moreover, the .lr_find() function, a feature designed to aid in the selection of an optimal learning rate, unfortunately led to confusion in my case. Each execution suggested an increment in the learning rate, which consistently resulted in overfitting and the subsequent degradation of the model's performance.

The journey through these frameworks has been enlightening, highlighting that the choice of a framework can often be dictated by the project requirements and personal familiarity. In the end, proficiency in any of these tools is a valuable asset in the field of data science and machine learning.

## Model Performance Comparison

Upon completion of the project, a comparison between Deep Learning (DL) and Machine Learning (ML) models was performed for the green taxi fare prediction task.

### Deep Learning Model Results
Mean Absolute Error (MAE): 0.674
Root Mean Squared Error (RMSE): 2.438
### Machine Learning Model Best Result
Mean Absolute Error (MAE): 0.576
Root Mean Squared Error (RMSE): 2.374
### Comparative Analysis
The results indicate that the Machine Learning model has a lower Mean Absolute Error and Root Mean Squared Error compared to the Deep Learning models, suggesting a more accurate and consistent performance. This is particularly noteworthy as it illustrates that traditional ML approaches can still outperform DL in certain scenarios, especially when the dataset is not exceedingly complex or large.

In this case, the ML model provided predictions that were, on average, closer to the actual fare amounts and demonstrated fewer large-scale errors. This could be attributed to various factors such as the models' architectures, the dataset's nature, the extent of hyperparameter optimization, and the volume of data utilized in training.

## Concluding Remarks

This project highlights the importance of context when choosing between ML and DL models. While DL models offer significant advantages for handling large and complex datasets, traditional ML models may be more suitable for tasks with smaller datasets or less complex prediction problems. The key takeaway is that the best model is not always the most complex one, but rather the one that is most appropriate for the data and task at hand.