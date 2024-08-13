import re
import string
import unicodedata
from collections import Counter

_CITATION = """\
@inproceedings{rajpurkar-etal-2016-squad,
    title = "{SQ}u{AD}: 100,000+ Questions for Machine Comprehension of Text",
    author = "Rajpurkar, Pranav  and
      Zhang, Jian  and
      Lopyrev, Konstantin  and
      Liang, Percy",
    booktitle = "Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2016",
    address = "Austin, Texas",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D16-1264",
    doi = "10.18653/v1/D16-1264",
    pages = "2383--2392",
}
@inproceedings{lee-etal-2019-latent,
    title = "Latent Retrieval for Weakly Supervised Open Domain Question Answering",
    author = "Lee, Kenton  and
      Chang, Ming-Wei  and
      Toutanova, Kristina",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P19-1612",
    doi = "10.18653/v1/P19-1612",
    pages = "6086--6096",
}
"""

_DESCRIPTION = """\
Exact match score for Open-domain Question Answering. 
This metric measures the percentage of predictions that match any one of the ground truth answers exactly.
"""

_KWARGS_DESCRIPTION = """
Calculates the percentage of predictions that match any one of the ground truth answers exactly.
Args:
    predictions: list of predictions to score. Each predictions
        should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each
        reference should be a list of strings with tokens separated by spaces.
Returns:
    em: description of the first score,
Examples:
    >>> em_metric = datasets.load_metric("exact_match")
    >>> results = em_metric.compute(references=[["apple", "orange"], ["banana"]], predictions=["apple", "pear"])
    >>> print(results)
    {'em': 0.5}
"""


def eval_generation_em(refs, preds):
    scores = []
    for ref, pred in zip(refs, preds):
        ref_answer = ref["answer"]
        em = metric_max_over_ground_truths(exact_match_score, pred, ref_answer)
        scores.append(em)
    avg_score = sum(scores) / len(scores)
    return avg_score


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def normalize_answer(s):
    """Normalize answer."""
    s = unicodedata.normalize("NFD", s)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def eval_generation_em_answers(refs, preds):
    scores = []
    for ref, pred in zip(refs, preds):
        ref_answer = ref["answers"]
        em = metric_max_over_ground_truths(exact_match_score, pred, ref_answer)
        scores.append(em)
    avg_score = sum(scores) / len(scores)
    return avg_score


def exact_match_score_with_multiple_candidates(prediction, ground_truths):
    pred = normalize_answer(prediction)
    for ground_truth in ground_truths:
        if pred == normalize_answer(ground_truth):
            return True
    return False

