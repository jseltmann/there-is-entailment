# there-is-entailment

This is a course project for the [Computational Semantics with Pictures](https://compling-potsdam.github.io/sose19-pm1-pictures/) course at Uni Potsdam.

We use image captions to try to predict the objects contained in the image. For that we combine captions from the [MS COCO dataset](http://cocodataset.org) with object annotations from the [Visual Genome dataset](http://visuagenome.org).

## True-False Classification
In this task, the model is given a pair consisting of a caption and an object. It then has to classify whether or not the object is likely to be in the image. The directory *class_true_false* contains lstm models to solve the task. The *baselines* directory contains a PMI based basedline for the task and code to train BERT for it.

## Generation task
Here the task is for the model to guess an object based on a given caption. *generation/one_obj/bert* contains the code to train BERT to do that.
