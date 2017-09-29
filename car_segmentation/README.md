# Kaggle Carvana Image Masking Challenge

This is my code using in the competition.

If any one wants to use it, it still needs some modification.

Some part of these codes come from [Keras starter](https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/37523).

Just lists the results of all model:

Model | Public leaderboard | image size
------|-----------------|---------|
Unet | 0.9963 | 1024x1024
Enet | 0.9962 | 512x512
LinkNet_Res34asEncoder | 0.994971 | 1024x1024

# Training Tips
## data augmentation
There are many kinds of augmentation methods.

However, I just use scale and shift for my best results.
Others would lower public scores.
I have tested all methods in my codes using 256x256 image size.

## batch size and gradient accumulator
I have used two gradient accumulators in utils.py, Adam and SGD.
I implemented SGD on my own because the code provided by another kaggler would not work.
And, the Keras version would affect on this. This came from Keras 2.0.8.

In the beginning, I hope it can help me to solve the memory issue of my GPU.
Definitely, it helps to train network more stably.

Nevertheless, there are still some limitations.

First, the batch size cannot be 1. It would still crush all process.

Second, the bigger batch size is not always the better choice.
The batch size still depends on dataset.

In conclusion, it is still useful for my results.

## image size
The results of this competition can be benefitted if using large dimension.
But obviously, it consumes a lot of time.

Accordingly, I start testing my training process from 256x256, such as data augmentation, optimizers, weighting boundary, networks, batch size and so on.

However, after I prepare all things and move to 1024x1024, all things are changed.

I just restart to tune my training process from scratch.

At the end, I have no enough time.

## some thing from this competition
I am too focus on networks and training process.

I think that I would start to do some feature engineering next time because it would be less affected by image size.
