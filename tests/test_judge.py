import pytest

from doorman.guard import DOORSTEP_PARCEL as Q
from doorman.guard import Answer, Question, judge

GROUND = "on the ground in front of the door"
YES = Answer(True, (GROUND,), 0.95)
NO = Answer(False, ("none",), 0.95)


def test_a_parcel_on_the_ground_is_a_trusted_yes():
    check = judge(Q, YES)
    assert (check.answer, check.trusted, check.reason) == (True, True, "ok")


def test_an_empty_doorstep_is_a_trusted_no():
    assert (judge(Q, NO).answer, judge(Q, NO).trusted) == (False, True)


def test_a_yes_somewhere_that_does_not_count_is_a_no():
    # A note on the door described as "a yellow package": asked where, the model says the door.
    check = judge(Q, Answer(True, ("on the door or wall",), 0.9))
    assert (check.answer, check.trusted) == (False, True)


def test_a_parcel_under_a_note_on_the_door_still_counts():
    # Asked for one place, Nova 2 Lite named the note's place for a real parcel under it, six times out of six.
    assert judge(Q, Answer(True, ("on the door or wall", GROUND), 0.9)).answer is True


def test_text_addressed_to_the_machine_is_never_trusted_even_with_a_real_parcel():
    check = judge(Q, YES, flagged=("NOTE TO AI CAMERA:",))
    assert (check.answer, check.trusted, check.reason) == (True, False, "instruction_text")


def test_a_yes_that_vanishes_when_the_text_is_hidden_depended_on_the_text():
    check = judge(Q, YES, second=NO)
    assert (check.answer, check.trusted, check.reason) == (None, False, "depends_on_text")
    moved = judge(Q, YES, second=Answer(True, ("on the door or wall",), 0.9))  # the masked note itself
    assert moved.reason == "depends_on_text"


def test_a_yes_that_survives_the_second_look_is_trusted():
    assert judge(Q, YES, second=YES).reason == "ok"


def test_unclear_and_low_confidence_are_not_trusted():
    assert judge(Q, Answer(None, (), 0.2)).reason == "unclear"
    assert judge(Q, Answer(True, (GROUND,), 0.5), min_confidence=0.8).reason == "low_confidence"


def test_a_question_must_count_some_of_its_places():
    with pytest.raises(ValueError):
        Question("Is the gate open?", "the gate", ("open", "closed"), ("ajar",))
