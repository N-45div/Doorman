import io

from PIL import Image, ImageDraw

from doorman.mask import mask_text


def frame_with_note() -> bytes:
    img = Image.new("RGB", (400, 200), (226, 214, 190))  # a wall
    d = ImageDraw.Draw(img)
    d.rectangle([100, 50, 300, 110], fill=(255, 238, 120))  # a yellow note
    d.text((110, 60), "PACKAGE INSIDE", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def pixel(image: bytes, x: int, y: int) -> tuple[int, int, int]:
    return Image.open(io.BytesIO(image)).convert("RGB").getpixel((x, y))


def test_text_regions_take_the_colour_around_them_not_grey():
    note_box = (100 / 400, 50 / 200, 200 / 400, 60 / 200)  # the whole note, as Textract would box a line
    masked = mask_text(frame_with_note(), [note_box])
    r, g, b = pixel(masked, 200, 80)
    assert abs(r - 226) < 12 and abs(g - 214) < 12 and abs(b - 190) < 12  # wall colour, not black text or grey


def test_a_text_box_inside_a_note_is_filled_with_the_note_colour():
    line_box = (108 / 400, 58 / 200, 100 / 400, 14 / 200)  # just the line of text
    masked = mask_text(frame_with_note(), [line_box])
    r, g, b = pixel(masked, 140, 65)
    assert r > 230 and g > 215 and b < 150  # the yellow of the note around it


def dark_pixels(image: bytes) -> int:
    img = Image.open(io.BytesIO(image)).convert("RGB")
    return sum(1 for x in range(100, 300) for y in range(50, 110) if sum(img.getpixel((x, y))) < 200)


def test_masking_removes_the_text_and_nothing_else_does():
    original = frame_with_note()
    assert dark_pixels(original) > 20
    assert dark_pixels(mask_text(original, [(100 / 400, 50 / 200, 200 / 400, 60 / 200)])) == 0
    for boxes in ([], [(0.5, 0.5, 0.0, 0.0)]):  # no boxes, and a box with no area
        assert dark_pixels(mask_text(original, boxes)) > 20
