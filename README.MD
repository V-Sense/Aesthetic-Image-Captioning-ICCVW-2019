 
 # Aesthetic Image Captioning from Weakly-Labelled Photographs
 ### Koustav Ghosal, Aakanksha Rana, Aljosa Smolic
 ###### Published at: [International Workshop on Cross-Modal Learning in Real World , ICCV 2019](https://cromol.github.io/)

PyTorch implementation for the [paper](https://arxiv.org/abs/1908.11310).
![alt text](https://v-sense.scss.tcd.ie/wp-content/uploads/2019/08/Screenshot-from-2019-08-26-15-59-18.png, "Results")


We provide json files in coco format and thus they can be used with any framework that uses the same. For our experiments, we use the [PyTorch implementation provided here.](https://github.com/ruotianluo/ImageCaptioning.pytorch.git)

Once the data is cleaned using the scripts provided, it can be plugged in to their code easily. Several web-crawlers are available for downloading the AVA Images. This data and the scripts should be sufficient to try the idea out on a different dataset. Hopefully, this page will be updated (blame procrastination if it doesn't) with cleaner and commented code and please check regularly. 

Until then, please contact ghosalk@tcd.ie for issues. 
##### Data

[Here's a link to Google Drive containing the following:](https://drive.google.com/drive/folders/1WPNUfUwk__fYF_P1pO4gSFXHRYjBwpCY?usp=sharing)

```
1. AVA_Comments_Full.txt : File contains the raw AVA_Comments with image IDs.
2. CLEAN_AVA_FULL_COMMENTS.json : A basic cleaning applied to the raw text data and saved as json
3. CLEAN_AVA_FULL_AFTER_SUBJECTIVE_CLEANING.json : AVA-Captions dataset with clean comments
4. AVA_TRAINING_FULL_DELIMITED_LDA_LABELLED.json: AVA-Captions json with LDA labels added for CNN training
5. LDA_AVA_200_50_passes_10000_iter_.p: LDA model trained on AVA-Captions dataset
6. _temp_topics_iteration_200.txt: Topic list corresponding to the model
7. AVA_FULL_VALIDATION.json: AVA-Captions validation set of 9362 images and corresponding captions. Format same as coco validation set.
```
##### Scripts

```
1. clean_comment_json.py : Performs a basic cleaning of the text file #1 and returns #2
2. clean_using_subjectivity.py: Implements the probabilistic caption filtering strategy on #2 and returns #3
3. lda_with_n_grams.py: Runs LDA with 200 topics on #3 and returns #5 and #6
4. topic_modelling_labelling.py: Labels #3 and returns #4
5. spice_best.py: Computes spice for all reference captions and returns the best. This isn't implemented in COCO evaluation code and should be added inside coco-caption/pycocoevalcap/spice/
```

## Bibtex
```
@article{ghosal2019aesthetic,
  title={Aesthetic Image Captioning From Weakly-Labelled Photographs},
  author={Ghosal, Koustav and Rana, Aakanksha and Smolic, Aljosa},
  journal={arXiv preprint arXiv:1908.11310},
  year={2019}
}
```





