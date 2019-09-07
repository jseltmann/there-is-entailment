import csv
import collections as col
import os


def object_overlap(tsv1, tsv2, log_path):
    """
    For two tsv files, compare which objects they contain.

    Parameters
    ----------
    tsv1 : str
        Path to first tsv file.
    tsv2 : str
        Path to second tsv file.
    log_path : str
        Path to write results to.
    """

    objs1 = col.defaultdict(int)
    objs2 = col.defaultdict(int)

    with open(tsv1) as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t', escapechar="\\")
        for i, line in enumerate(reader):
            if i == 0:
                continue
            obj = line[2]
            objs1[obj] += 1
            
    with open(tsv2) as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t', escapechar="\\")
        for i, line in enumerate(reader):
            if i == 0:
                continue
            obj = line[2]
            objs2[obj] += 1

    inters = dict()
    for obj in objs1:
        if obj in objs2:
            inters[obj] = (objs1[obj], objs2[obj])

    examples_overall1 = 0
    examples_not_inters1 = 0
    for obj in objs1:
        examples_overall1 += objs1[obj]
        if obj not in objs2:
            examples_not_inters1 += objs1[obj]

    examples_overall2 = 0
    examples_not_inters2 = 0
    for obj in objs2:
        examples_overall2 += objs2[obj]
        if obj not in objs1:
            examples_not_inters2 += objs2[obj]


    log_dir = log_path[:-4]
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    with open(log_dir + "/objs1.log", "w") as of1:
        objs = sorted(objs1.items(), key=(lambda x: x[1]))
        for obj in objs:
            of1.write(str(obj) + "\n")
    with open(log_dir + "/objs2.log", "w") as of1:
        objs = sorted(objs2.items(), key=(lambda x: x[1]))
        for obj in objs:
            of1.write(str(obj) + "\n")
    with open(log_dir + "/objs_comb.log", "w") as of1:
        objs = sorted(inters.items())
        for obj in objs:
            of1.write(str(obj) + "\n")

    with open(log_path, "w") as log_file:
        log_file.write("file 1: " + tsv1 + "\n")
        log_file.write("objects in 1: " + str(len(objs1)) + "\n")
        log_file.write("examples in 1: " + str(examples_overall1) + "\n")
        log_file.write("examples exclusively in 1: " + str(examples_not_inters1) + "\n")
        log_file.write("\n\n")
        log_file.write("file 2: " + tsv2 + "\n")
        log_file.write("objects in 2: " + str(len(objs2)) + "\n")
        log_file.write("examples in 2: " + str(examples_overall2) + "\n")
        log_file.write("examples exclusively in 2: " + str(examples_not_inters2) + "\n")
        log_file.write("\n\n")
        log_file.write("objects in common: " + str(len(inters)) + "\n")
    


object_overlap("../../data/bert_classify_thereis_5caps_seed0/train.tsv",
               "../../data/bert_classify_thereis_5caps_seed0/dev.tsv",
               "../../logs/data_analysis/object_overlap.log")
