# pattern-recognition
Project to advances in pattern recognition and tecnical experiences

Principal Results:
Metrics choosed of classification:
  I choosed to use as classification metrics, the [accuracy, precision, f1-score, recall, precision], which has the characteristic:
  - accuracy: Describes how the model performs across all the classes
  - precision: The precision measures the model's accuracy in classifying a sample as positive.
  - recall: Quantifies the number of correct positive predictions made out of all positive predictions
  - f1-score: Combines the precision and recall, and give a  harmonic mean of them
  This metrics are presented in each jupyter notebook.
  
1. Using the Bag of Visual Words:
The results in this approach showed that the number of visual words is very important. When I utilized a small number, like 2 or 5, the confusion matrix dont´t
showed convergency. 
![results](https://user-images.githubusercontent.com/65249438/204516693-3cc9f285-a1fc-45de-97df-a9b5ca90ac7e.png)
As the value was increased, an increase in performance was seen, with a lot of computational cost, but the models takes a long time to converge, having horrible 
results in this approach

Due to this, feature extraction with vgg16 showed much better results.

2. Using the vgg16 cnn as features exctator:
The results of this case are great. Each class of dataset has a good metric

![image](https://user-images.githubusercontent.com/65249438/204488555-d8064a78-eac4-4cf6-8c28-23bd45fa299d.png)

---
## How to run
1. CLone the project
2. Install the poetry
3. Execute:
  ```poetry init```
  ```poetry shell```

4. Now, you can open the jupyter notebook and execute
