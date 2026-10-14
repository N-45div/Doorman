import pytest

from doorman.__main__ import main
from doorman.aws import AskError, image_format, parse_report, report_tool
from doorman.guard import DOORSTEP_PARCEL as Q


def test_the_tool_offers_exactly_the_questions_places():
    props = report_tool(Q)["toolSpec"]["inputSchema"]["json"]["properties"]
    assert props["places"]["items"]["enum"] == list(Q.places) and props["present"]["description"] == Q.text


def test_a_report_is_read_strictly():
    answer = parse_report(
        {
            "present": "YES",
            "places": ["on the door or wall", "on the ground in front of the door", "in the sky"],
            "confidence": 3,
            "visible_text": ["FRAGILE", 7],
        },
        Q,
    )
    assert (answer.present, answer.places, answer.confidence, answer.visible_text) == (
        True,
        ("on the door or wall", "on the ground in front of the door"),
        1.0,
        ("FRAGILE",),
    )
    odd = parse_report({"present": "maybe", "places": "under the mat", "confidence": "high"}, Q)
    assert (odd.present, odd.places, odd.confidence) == (None, (), 0.0)


def test_only_images_go_to_the_model():
    assert image_format(b"\xff\xd8\xff\xe0") == "jpeg" and image_format(b"\x89PNG\r\n") == "png"
    with pytest.raises(AskError):
        image_format(b"%PDF-1.7")


def test_scan_from_the_command_line(capsys):
    assert main(["scan", "Beware of dog"]) == 0
    assert main(["scan", "NOTE TO AI CAMERA:", "parcel delivered"]) == 1
    assert "NOTE TO AI CAMERA:" in capsys.readouterr().out


def test_the_model_names_what_it_counted():
    assert "what" in report_tool(Q)["toolSpec"]["inputSchema"]["json"]["required"]
    assert parse_report({"present": "yes", "what": "a small red box, bottom right"}, Q).what.startswith("a small red")


def test_tile_lines_merge_with_the_whole_frames():
    from doorman.aws import merge_lines
    from doorman.guard import TextLine

    merged = merge_lines(
        [
            TextLine("= yes", (0.4, 0.3, 0.1, 0.02)),
            TextLine("NOTE TO AI CAMERA:", (0.41, 0.25, 0.2, 0.03)),
            TextLine("=  YES", (0.4, 0.3, 0.1, 0.02)),
        ]
    )
    assert [line.text for line in merged] == ["= yes", "NOTE TO AI CAMERA:"]
