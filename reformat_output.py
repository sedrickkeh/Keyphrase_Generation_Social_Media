import os
import argparse
import json

def reformat(task, experiment, steps):
    preds = []
    filepath = "output/{}/pred/{}_step_{}/{}.pred".format(experiment, task, steps, task)
    for line in open(filepath, 'r'):
        output = json.loads(line)
        pred_sents = output["pred_sents"]
        pred = ""
        for sent in pred_sents:
            hashtag = ""
            for word in sent: hashtag = hashtag + word + " "
            hashtag = hashtag.strip()
            pred = pred + ";" + hashtag
        preds.append(pred[1:])

    savepath = "output/{}/pred/{}_step_{}/{}_processed.pred".format(experiment, task, steps, task)
    with open(savepath, 'w') as f:
        for pred in preds:
            f.write("%s\n" % pred)


def main(config):
    task = config.task.lower()
    experiment = config.experiment.lower()
    steps = config.steps
    reformat(task, experiment, steps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="twitter_conv")
    parser.add_argument('--experiment', type=str, default="twitter_conv_0")
    parser.add_argument('--steps', type=str, default="10000")
    config = parser.parse_args()

    main(config)
