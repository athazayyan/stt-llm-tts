import json
import os


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


JSON_PATH = os.path.join(

    BASE_DIR,

    "..",

    "data",

    "manifest",

    "transcripts.json"

)


OUTPUT_DIR = os.path.join(

    BASE_DIR,

    "..",

    "data",

    "corpus",

    "transcripts"

)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


with open(
    JSON_PATH,
    encoding="utf8"
) as f:

    transcripts = json.load(f)



for audio_id,text in transcripts.items():

    save_path = os.path.join(

        OUTPUT_DIR,

        f"{audio_id}.txt"

    )


    with open(

        save_path,

        "w",

        encoding="utf8"

    ) as txt:

        txt.write(text)


print(
    "Done generating transcripts"
)