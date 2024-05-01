import re
import mock
import os
from pytube.cipher import get_throttling_function_code
from pytube import YouTube
import whisper
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache
from langchain_core.output_parsers import StrOutputParser
from openai import OpenAI


def patched_throttling_plan(js: str):
    """Patch throttling plan, from https://github.com/pytube/pytube/issues/1498"""
    raw_code = get_throttling_function_code(js)

    transform_start = r"try{"
    plan_regex = re.compile(transform_start)
    match = plan_regex.search(raw_code)

    #transform_plan_raw = find_object_from_startpoint(raw_code, match.span()[1] - 1)
    transform_plan_raw = js

    # Steps are either c[x](c[y]) or c[x](c[y],c[z])
    step_start = r"c\[(\d+)\]\(c\[(\d+)\](,c(\[(\d+)\]))?\)"
    step_regex = re.compile(step_start)
    matches = step_regex.findall(transform_plan_raw)
    transform_steps = []
    for match in matches:
        if match[4] != '':
            transform_steps.append((match[0],match[1],match[4]))
        else:
            transform_steps.append((match[0],match[1]))

    return transform_steps


with mock.patch('pytube.cipher.get_throttling_plan', patched_throttling_plan):
    from pytube import YouTube

    url = input('Input youtube URL here, type "test" to run test URL\n')

    if url.lower() == 'test':
        url = 'https://www.youtube.com/watch?v=VIR46oH-ufk&ab_channel=Horses'

    video = YouTube(url)
    audio = video.streams.filter(only_audio=True, file_extension='mp4')[0]

    
    dir = video.title
    check_dir = os.path.isdir(dir)
    if not check_dir:
        os.mkdir(video.title)
    else:
        pass

    audio.download(filename=video.title + '/' + video.title + '.mp4')


model = whisper.load_model("tiny.en")
text = model.transcribe(video.title + '/' + video.title + '.mp4', fp16=False)

with open(video.title + '/' + video.title + '.txt' , 'w') as f:
    f.write(text["text"])
    f.close()



content = open(video.title + '/' + video.title + '.txt', 'r').read()

set_llm_cache(SQLiteCache(database_path=".langchain.db"))
output_parser = StrOutputParser()


model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)

prompt = PromptTemplate(
    template="""Create 2 separate paragraphs based on this text, start the first paragraph with the words /
    'The idea of this text...' and try to summarize it into at least 10-15 sentences,/
    in second paragraph give 4 simple statements in numbered list form based on of this text. Text ====== {text}""",
    input_variables=["text"],
)


chain = prompt | model | output_parser

with open(video.title + '/' + 'AI_' + video.title + '.txt', 'w') as f:
    f.write(chain.invoke(content))
    f.close()


audio_req = input('Create audio file?: [y/n] ')
    
if audio_req == 'y':
    client = OpenAI()
    response = client.audio.speech.create(
        model = 'tts-1',
        voice = 'echo',
        input = open(video.title + '/' + 'AI_' + video.title + '.txt', 'r').read()
    )
    response.stream_to_file(video.title + '/' + 'AI_' + video.title + '.mp4')