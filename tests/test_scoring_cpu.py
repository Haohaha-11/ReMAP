import torch

from remap.scoring.answers import score_answer
from remap.scoring.benchmarks import can_infer, mimic_score, score_prediction, typed_final_payload
from remap.training.rl.rewards import RewardSettings, shaped_reward
from remap.training.stage1.roi import box_loss, coverage_target, enclosing_boxes

MCQ = {"answer_type": "public_mcq", "gold_answer": "A", "acceptable_answers": ["A"],
       "public_options": {"A": "None of the choices provided", "B": "6", "C": "4"}, "correct_letter": "A"}


def test_typed_choice_answers():
    assert score_answer("reasoning\nAnswer: A", MCQ, n_images=2).correct
    assert score_answer("reasoning\nAnswer: (A) None of the choices provided", MCQ, n_images=2).correct
    assert not score_answer("reasoning\nAnswer: B", MCQ, n_images=2).correct
    # Two different commitments are a conflict, not a correct answer.
    assert not score_answer("Answer: A\nAnswer: B", MCQ, n_images=2).parsed


def test_integer_answers_with_unit():
    reward = {"answer_type": "integer", "gold_answer": "3"}
    question = ("How many dog instances are there in total across these 2 images? "
                "Count objects, including multiple objects in the same image.")
    assert score_answer("Answer: 3", reward, n_images=2, question=question).correct
    assert score_answer("Answer: 3 dogs", reward, n_images=2, question=question).correct
    assert not score_answer("Answer: 3 images", reward, n_images=2, question=question).parsed


def test_shaped_reward():
    settings = RewardSettings()
    assert shaped_reward(1.0, 0, 0.0, settings)[0] == 1.0
    assert abs(shaped_reward(1.0, 1, 0.0, settings)[0] - 1.12) < 1e-12
    assert abs(shaped_reward(0.0, 2, 0.0, settings)[0] + 0.002) < 1e-12
    # Bonus decays to a quarter at the end of training; only three reads earn it.
    assert abs(shaped_reward(1.0, 1, 1.0, settings)[1] - 0.03) < 1e-12
    assert shaped_reward(1.0, 4, 0.0, settings)[0] < shaped_reward(1.0, 3, 0.0, settings)[0]


def test_benchmark_parsers():
    choices = {"A": "rubber", "B": "cotton"}
    assert can_infer("The glove is rubber.\nAnswer: A", choices) == "A"
    assert typed_final_payload("x\nAnswer: skiing") == ("skiing", "answer_payload")
    assert mimic_score("mimic_counting", "3", "three")[1]
    record = score_prediction("mimic_listing", "Items: laptop, fridge\nAnswer: laptop, fridge", "laptop, fridge")
    assert record["payload_score"] == 1.0 and record["score"] < 1.0    # main metric uses the raw output


def test_roi_loss_prefers_covered_cells():
    target = coverage_target([0.0, 0.0, 0.5, 0.5], (4, 4), device="cpu")
    assert target.sum() == 4
    good = torch.full((4, 4), -4.0)
    good[:2, :2] = 4.0
    assert box_loss(good, [0.0, 0.0, 0.5, 0.5]) < box_loss(-good, [0.0, 0.0, 0.5, 0.5])
    boxes = enclosing_boxes([{"image_index": 0, "bbox_normalized": [0.1, 0.1, 0.2, 0.2]},
                             {"image_index": 0, "bbox_normalized": [0.5, 0.5, 0.6, 0.7]}], 2)
    assert boxes == {0: [0.1, 0.1, 0.6, 0.7]}
