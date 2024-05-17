Business needs

    Using the video game sales dataset, we can predict behavior to retain customers.
    The Global_Sales rate in this context describes the number of players who stop buying video games over a
    certain period of time. Each game is assigned a predictive value that estimates the probability of
    its exit at any point in time.
    
Requirements

    python 3.7

    numpy==1.17.3
    pandas==1.1.5
    sklearn==1.0.0

Running:

    To run the demo, execute:
        python predict.py 

    After running the script in that folder will be generated prediction_results.csv 
    The file has 'Global_Sales_pred' column with the result value.

    The input is expected  csv file in the same folder with a name new_input.csv. The file shoul have all features columns. 

Training a Model:

    Before you run the training script for the first time, you will need to create dataset. The file train.csv should contain all features columns and target for prediction Global_Sales.
    After running the script the "finalized_model.saw" will be created.
    Run the training script:
        python train.py

    The model mean absolute error is 0.011
    The model mean squared error is 0.46
    The model root mean squared error is 0.86
    There is no fraud check.