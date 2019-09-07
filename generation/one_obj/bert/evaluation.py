import pickle
import csv

def generate_readable_file(eval_out_filename, dev_filename, out_dir):
    """
    Take the numeric outputs generated by Bert and match them with the examples
    in the dev file.

    Parameters:
    -----------
    eval_out_filename : str
        Path to pickle file containing the bert results.
    dev_filename : str
        Path to file containing the original examples.
    out_dir : str
        Path to write the results to.
    """

    with open(eval_out_filename, "rb") as eval_file:
        preds, label_ids, label_map = pickle.load(eval_file)

    rev_map = {v:k for (k,v) in label_map.items()}
    corr = 0
    false = 0

    with open(dev_filename) as dev_file:
        with open(out_dir + "/corr.log", "w") as corr_file:
            with open(out_dir + "/false.log", "w") as false_file:
                reader = csv.reader(dev_file, delimiter='\t', escapechar="\\")
                for i, line in enumerate(reader):
                    if i == 0:
                        continue
                    ii, cap, _ = line
                    pred = preds[i-1]
                    pred_obj = rev_map[pred]
                    label_id_set = set(label_ids[i-1])
                    objects = [rev_map[lid] for lid in label_id_set]
                    out_line = ii + "\t" + cap + "\t" + str(objects) + "\t" + pred_obj + "\n"
                    if pred in label_id_set:
                        corr_file.write(out_line)
                        corr += 1
                    else:
                        false_file.write(out_line)
                        false += 1
    
    acc = corr / (corr + false)
    with open(out_dir + "/accuracy.log", "w") as out_file:
        out_file.write("accuracy: " + str(acc) + "\n")


eval_out_filename = "/home/users/jseltmann/there-is-entailment/logs/generation/one_obj/bert_logs/eval_easy_output.pkl"
dev_filename = "/home/users/jseltmann/there-is-entailment/data/generation_data_seed0_harder/dev.tsv"
out_dir = "/home/users/jseltmann/there-is-entailment/logs/generation/one_obj/bert_logs/"

generate_readable_file(eval_out_filename, dev_filename, out_dir)
            
