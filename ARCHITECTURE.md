# Doorman architecture

How one check works, and how the pieces fit. The [README](README.md) covers what Doorman is for and how to use it.

## 1. The modules

```mermaid
flowchart LR
    G["guard.py: Question, Answer, Check, Doorman, judge()"] --> R["rules.py: scan() for text addressed to a machine"]
    G --> MK["mask.py: mask_text() paints text out"]
    G -.->|"Ocr protocol"| AWS["aws.py: TextractOcr, ConverseAsker"]
    G -.->|"Asker protocol"| AWS
    CLI["__main__.py: python -m doorman scan / check"] --> G
    EV["eval.py: python -m doorman.eval"] --> G
    EV --> AWS
```

`guard.py` depends only on two small protocols, `Ocr.read(image)` and `Asker.ask(image, question)`. `aws.py` implements them with Amazon Textract and any Bedrock vision model through the Converse API; any other OCR or model plugs in the same way. `rules.py` and `mask.py` need no cloud at all.

## 2. One check

```mermaid
sequenceDiagram
    autonumber
    participant C as Caller
    participant D as Doorman.check
    participant O as Ocr (Textract)
    participant A as Asker (Bedrock model)
    participant J as judge()

    C->>D: check(image, question)
    D->>O: read(image)
    O-->>D: lines of text, each with its box
    D->>A: ask(image, question)
    A-->>D: present, places, what, confidence, visible text
    D->>D: scan the OCR lines and the model's own text, separately
    alt nothing flagged, the answer counts as yes, and the frame holds text
        D->>D: paint every text box with the colour around it
        D->>A: ask(masked image, question)
        A-->>D: second answer
    end
    D->>J: question, first answer, flagged lines, second answer
    J-->>C: Check: answer, trusted, reason, flagged, both answers, OCR text
```

The second look is only taken when it can change the outcome: the first answer is a yes in a counted place, nothing was flagged, and there is text to paint out.

## 3. How judge() decides

```mermaid
flowchart TB
    S["first answer, flagged lines, second answer"] --> F{"Any line addressed to a machine?"}
    F -->|"yes"| U1["not trusted · instruction_text"]
    F -->|"no"| Y{"Yes in a counted place, but not after the text was painted out?"}
    Y -->|"yes"| U2["not trusted · depends_on_text"]
    Y -->|"no"| UN{"Answer unclear?"}
    UN -->|"yes"| U3["not trusted · unclear"]
    UN -->|"no"| LC{"Confidence below the minimum?"}
    LC -->|"yes"| U4["not trusted · low_confidence"]
    LC -->|"no"| OK["trusted · ok"]
```

A yes only counts when one of the places the model named is in `question.counts`. A note on a door described as a parcel is placed "on the door or wall", which does not count, so the answer becomes no. `judge()` does no I/O, so every rule is tested on its own.

## 4. Reading text from a camera frame

```mermaid
flowchart LR
    IMG["Frame"] --> W["Textract: whole frame"]
    IMG --> T1["Textract: top-left 60%"]
    IMG --> T2["Textract: top-right 60%"]
    IMG --> T3["Textract: bottom-left 60%"]
    IMG --> T4["Textract: bottom-right 60%"]
    T1 & T2 & T3 & T4 --> MAP["map each box back to the whole frame"]
    W --> MERGE["merge: one line per text"]
    MAP --> MERGE
    MERGE --> LINES["lines with boxes"]
```

Textract reads documents, and on a whole camera scene it can skip lines: on a frame with a parcel under a three-line note it returned only the last line. The four overlapping quarters give it the text at document scale. The five reads run in parallel.

## 5. The eval

```mermaid
flowchart TB
    CASES["30 drawn cases: 20 planted notes on an empty doorstep, 6 ordinary notes, 4 controls"] --> RENDER["render each frame, deterministic"]
    RENDER --> OCR["OCR each frame once"]
    OCR --> RUN["for each model: first answer, flags, second answer on any yes"]
    RUN --> LAYERS["score what each layer removes from one run"]
    LAYERS --> TABLE["table + evals/results.json"]
```

Every run records the model's first answer, what the rules flagged and the second look, so the table shows the model alone, then each layer added, without rerunning the model per layer. Controls check that a real parcel still comes through and that a real parcel under a note to the AI is never trusted.
