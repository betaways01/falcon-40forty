import json
import os
import argparse

# the following dictionary is a simplified version of the original SUBCATEGORIES dictionary for illustration
SUBCATEGORIES = {
  "STEM": ["machine_learning", "computer_science"],
  "all": ["machine_learning", "computer_science"],
}

def load_predictions_file(file):
  predictions = {}
  for line in open(file):
    dp = json.loads(line)
    # process the dp dictionary as per your requirements
  return predictions

def load_gold_file(file):
  gold = {}
  for line in open(file):
    dp = json.loads(line)
    # process the dp dictionary as per your requirements
  return gold

def score_categories(gold_answers, predictions, categories):
  acc = []
  for cat in categories:
    preds = predictions[cat]
    golds = gold_answers[cat]
    for question in golds.keys():
      pred = preds[question]
      gold = golds[question]
      acc.append(pred["prediction"] == gold)
  acc = sum(acc) / len(acc)
  return acc

def main(predictions_file, gold_file):
  print(f"predictions for {predictions_file}")
  print(f"{'category': >15}\t{'Acc(%)':>15}")
  predictions = load_predictions_file(predictions_file)
  gold_answers = load_gold_file(gold_file)
  print("-" * 30)
  for category_name, categories in SUBCATEGORIES.items():
    scores = score_categories(gold_answers, predictions, categories)
    sc = f"{100*scores:0.2f}"
    print(f"{category_name: >15}\t{sc:>15}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--predictions_path", type=str, help="Path to the written predictions file")
  parser.add_argument("--gold_path", type=str, help="Path to the gold data file")
  args = parser.parse_args()
  main(args.predictions_path, args.gold_path)
