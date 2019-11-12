# Cog-TAIL
## Conditional Generation of Text with Generative Adversarial Imitation Learning



### Preprocessing

#### 1. Introduction

Not important.



#### 2. Top3600 Words

The following is detail of pickle files.

- **Pre_reviews** : It consists of preprocessed Amazon electronic products review files. The original can be found at [here](http://jmcauley.ucsd.edu/data/amazon/?fbclid=IwAR33rEDsJtvy6ClQ6T4xZiOIT2o76eY39pGpT6P_1DTr9tD1HfRFFF_ixa).
- **Top3600_dist** : We use only 3600 words that appear frequently in learning. This file contains the distribution of them.
- **Top3600_AtoB**, **Top3600_BtoA** : Each word in the Top3600 has a unique index, which matches one-to-one with the result of the BERT encoding. This index can be used to make the first word, to see what word it was in a batch, and so on. AtoB converts from a unique index to a BERT index, while BtoA does the opposite.
- **Top3600_first_pos, Top3600_first_neg** : The cumulative distribution of the first word for each emotion. Use it to pick the first word of a sentence.



#### 3. Batch File Links

Each batch file has 5 different keys.

- **states** : BERT embedded sentence sequence, but just one index.
- **actions** : The word just after the sentence used to create **states**.
- **codes** : The sentiment of the sentence.
- **action_ids** : Top3600 index of the action.
- **prev_action_ids** : Top3600 index of the previous action.



Batch file created differently depending on the index used when creating **states**.

- [CLS](https://drive.google.com/open?id=13VTc6buQQfdmp0slqetIp3ZpqseRygx8)
- [The Word just before SEP](https://drive.google.com/open?id=1YY8sBK8QnP-jlZ5Z767f1qP5Jp9yDcgK)
- [SEP](https://drive.google.com/open?id=1FHMiQl0cXCRrq2WM7-xAJ58pjI1ODOad)