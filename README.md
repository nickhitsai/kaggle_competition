# Planet: Understanding the Amazon from Space

##This repo provide a Keras solution that can make you enter top 100 scores.

Just lists the results of all model:

Model | Public F2 score
inceptionV3 | 0.922
VGG16 | 0.928
DenseNet121 | 0.926
DenseNet121 | 0.925


# Training Tips

## Learning rate
In all files, all include a learning rate scheduler function for tuning learning rate.
It just based on my experiments.
I tune it from auto decrease callback(ReduceLROnPlateau)[https://keras.io/callbacks/].
I run the traning process about 40 epoch at the begining. I figure out that is too much for this competition.
After the first model, I just do this about 20 epoch.

On the other hands, I would keep training at a relatively low learning rate.
If there is a model that recall score is pretty high, that would be a big help on F2 score.


## Batch size
In my opinion, the bigger the better.
This parameter can help you reduce the variance of every batch.
If it is too small, it would crash all effort you have done.

Just provide some example.
If I train the same flow in these files, just change the batch_size to 4 or 8, the results would not reach the results I provide.


## Custom loss function
In this competition, it is scored by f2 score. I have also experimented the custom loss function.
Briefly, It's suck.
I have tried all kinds af combination, such as BinaryLoss + a\*F2Score(a locate in 0~1) and just F2Score.
All results are worse than BinaryLoss.
