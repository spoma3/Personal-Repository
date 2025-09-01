import speech_recognition as sr
import pyttsx3
import time
import datetime
import webbrowser
from ddgs import DDGS
from bs4 import BeautifulSoup
import requests
import nltk
from nltk.tokenize import sent_tokenize
import os
import re
import torch
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering
import multiprocessing as mp
import platform
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

if (platform.system()=='Darwin'):
    mp.set_start_method('spawn', force=True)
else:
    mp.set_start_method('fork', force=True)
recognizer = sr.Recognizer()

def speak(text):
    print(f"Speaking: {text}")
    temp_engine = pyttsx3.init()
    temp_engine.setProperty('volume', 0.7)
    temp_engine.say(text)
    temp_engine.runAndWait()
    temp_engine.stop()

def calibrate_ambient_noise_for_transcription():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration = 0.5)
        recognizer.dynamic_energy_threshold = True
        recognizer.dynamic_energy_adjustment_damping = 0.15

def listen():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration = 0.5)
        recognizer.dynamic_energy_threshold = True
        recognizer.dynamic_energy_adjustment_damping = 0.15
        print("Listening...")
        try:
            audio = recognizer.listen(source, timeout=None, phrase_time_limit=60)
            query = recognizer.recognize_google(audio)
            print(f"You said {query}")
            return query.lower()
        except sr.WaitTimeoutError:
            print("Listening timed out while waiting for phrase.")
        except sr.UnknownValueError:
            print("Sorry I did not understand that.")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition.")
            return ""

def transcribe():
    with sr.Microphone() as source:
        print("Listening...")
        try:
            audio = recognizer.listen(source, timeout=None, phrase_time_limit=60)
            statement = recognizer.recognize_google(audio)
            return statement.lower()
        except sr.WaitTimeoutError:
            print("Transcription timed out while waiting for phrase.")
        except sr.UnknownValueError:
            print("Sorry I did not understand that.")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition.")
            return ""

def to_grimdark(text):
    replacements = {
        r'\bgovernment\b' : 'Imperium',
        r'\benemy\b' : 'xenos threat',
        r'\bspace\b' : 'cold void',
        r'\bscience\b' : 'forbidden knowledge',
        r'\btechnology\b' : 'arcane machines',
        r'\brobot\b' : 'servitor',
        r'\bsoldier(s)?\b' : r'Astartes\1',
        r'\bleadership\b' : 'High Lords of Terra',
        r'\bdanger\b' : 'heresy',
        r'\bbrutes\b' : 'orks',
        r'\bbrutish\b' : 'orkish',
        r'\barrogant\b' : 'Eldar',
        r'\bsoulless\b' : 'Necrons',
        r'\babomination\b' : 'Necron like',
        r'\bparasite(s)?\b' : r'Tyranid\1',
        r'\binsidious\b' : 'Tau',
        r'\bidealistic\b' : 'Tau',
        r'\bhorrors\b' : 'chaos demons',
        r'\bdisgusting\b' : 'chaos demons',
        r'\bchaos\b' : 'ultimate heresy',
    }
    for pattern, grim_term in replacements.items():
        text = re.sub(pattern, grim_term, text, flags=re.IGNORECASE)
    return text

def load_models():
    bert_model_name = "deepset/bert-base-cased-squad2"
    bert_model = AutoModelForQuestionAnswering.from_pretrained(bert_model_name)
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name, legacy = False)
    qa_pipeline = pipeline("question-answering", model = bert_model, tokenizer = bert_tokenizer)

    gpt2_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    gpt2_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    generator = pipeline("text-generation", model = gpt2_model, tokenizer = gpt2_tokenizer)
    return qa_pipeline, generator, gpt2_tokenizer

def generate_response(prompt, qa_pipeline, generator, gpt2_tokenizer):
    try:
        results = list(DDGS().text(query=prompt, max_results=5))
        if not results:
            speak("I couldn't find anything on that.")
            return
        all_context = ""
        for result in results:
            url = result.get('href')
            if not url:
                continue

            try:
                res = requests.get(url, timeout=10)
                soup = BeautifulSoup(res.text, "html.parser")
                paragraphs = soup.find_all("p")
                info = " ".join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 40])
                budget = 200
                info = " ".join(info.split()[:budget])
                all_context += " " + info

            except requests.RequestException:
                continue

            if len(all_context.split()) > 1500:
                break
        if not all_context.strip():
            speak("I couldn't extract enough useful information.")
        else:
            simplified_context = " ".join(all_context.split()[:1000])
            answer = qa_pipeline(question=prompt, context= simplified_context)
            if (answer['score'] > 0.3):
                prompt = prompt + "? " + f"Here is the answer: {answer}"
                tokenized = gpt2_tokenizer(all_context, return_tensors="pt")
                if (tokenized.input_ids.shape[1] > 512):
                    all_context = gpt2_tokenizer.decode(tokenized.input_ids[0, -512:], skip_special_tokens=True)
                summarized = generator(all_context, max_new_tokens=128, do_sample=True, truncation=True)
            else:
                qa_pipeline, generator, gpt2_tokenizer = load_models()
                all_context = prompt + "? " + all_context
                tokenized = gpt2_tokenizer(all_context, return_tensors="pt")
                if (tokenized.input_ids.shape[1] > 512):
                    all_context = gpt2_tokenizer.decode(tokenized.input_ids[0, -512:], skip_special_tokens=True)
                summarized = generator(all_context, max_new_tokens=128, do_sample=True, truncation=True,)
            result = summarized[0]['summary_text']
            return result
    except Exception as e:
        return f"Error: {e} occurred in generate response."

def main_assistant(qa_pipeline, generator, gpt2_tokenizer):
    speak("Can begin cogitation now, what assistance is needed?")
    while True:
        command = listen()
        print("Command received: ", repr(command))
        if command is None:
            continue
        elif "time" in command:
            hour = datetime.now().strftime("%I").lstrip("0")
            minute = datetime.now().strftime("%M").lstrip("0")
            period = datetime.now().strftime("%p")
            speak(f"The current time is {hour} {minute} {period}")
        elif "date" in command:
            date = datetime.date.today()
            speak(f"The current date is {date}")
        elif any(phrase in command for phrase in ['start log', 'create log', 'make log', 'new log',
                                                   'start transcription', 'begin transcription', 'commence transcription', 'create transcription', 'new transcription']):
            speak("Creating Imperial log now")
            date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            try:
                log_dir = "Logs"
                os.makedirs("Logs", exist_ok = True)
                file_name = os.path.join(log_dir, f"{str(date)}.txt")
                with open(file_name, 'w') as f:
                    calibrate_ambient_noise_for_transcription()
                    while True:
                        try:
                            text = transcribe()
                            if (text is None):
                                continue
                            elif any(phrase in text for phrase in ['stop log', 'end log', 'finish log', 'finish transcription', 'end transcription', 'stop transcription']):
                                break
                            f.write(text + "\n")
                            f.flush()
                        except Exception as e:
                            print(f"Error {e} has occurred continuing transcription")
                    print(f"Log written to file path {date}")
            except IOError as e:
                print(f"Error creating file: {e}")
        elif "search" in command:
            command = command.replace("search ", "")
            command += "?"
            results = list(DDGS().text(command, max_results=1))
            if not results:
                speak("I couldn't find anything on that.")
                return
            url = results[0]['href']
            print(f"Found url: {url}")
            try:
                res = requests.get(url, timeout=10)
                soup = BeautifulSoup(res.text, "html.parser")
                paragraphs = soup.find_all("p")
                combined_text = " ".join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 30])
                try:
                    model_name = "t5-small"
                    tokenizer = T5Tokenizer.from_pretrained(model_name)
                    model = T5ForConditionalGeneration.from_pretrained(model_name)
                    summarizer = pipeline("summarization", model = model, tokenizer = tokenizer)
                    while True:
                        speak("How long do you want the summary to be")
                        word_length = listen()
                        if (word_length is None):
                            continue
                        number_string = ''.join(re.findall(r'\d+', word_length))
                        if (number_string is None or len(number_string) == 0):
                            continue
                        else:
                            break
                    number = int(''.join(re.findall(r'\d+', word_length)))
                    words = combined_text
                    summary = summarizer(words, max_length = max(50, min(number, 400)), min_length = 10, do_sample=True)[0]['summary_text']
                    summary = to_grimdark(summary)
                    speak(f"Here is a summary {summary}")
                except Exception as e:
                    speak(f"Error {e} occurred when loading summarization model")
            except Exception as e:
                print(f"Searched at {url}")
                speak("There was a problem reading the webpage")
        elif any(phrase in command for phrase in ['summarize log', 'summarize transcription', 'summarize Imperial log']):
            files = sorted(os.listdir("Logs"))
            files = [f for f in files if os.path.isfile(os.path.join("Logs", f))]
            if (len(files) == 1):
                number = 1
            elif (len(files) == 0):
                print("No Imperial Logs currently present for summarization")
            else:
                print(f"The list of Imperial Logs is: {files}")
                while True:
                    speak("Which file would you like")
                    file_text = listen()
                    if (file_text is None):
                        continue
                    number_map = {"one" : "1", "two" : "2", "three" : "3", "four" : "4", "five" : "5", "six" : "6", "seven" : "7", "eight" : "8", "nine" : "9", "last" : "0"}
                    string_index = [number_map[f] for f in file_text.split() if f in number_map]
                    if len(string_index) == 0:
                        continue
                    else:
                        break
                number = int(''.join(string_index)) - 1
                if (number > len(files)):
                    number = number%10
                search_file = files[number]
                with open(os.path.join("Logs", search_file), 'r') as f:
                    info = f.read()
                info = info.strip().lower()
                try:
                    model_name = "t5-small"
                    tokenizer = T5Tokenizer.from_pretrained(model_name)
                    model = T5ForConditionalGeneration.from_pretrained(model_name)
                    summarizer = pipeline("summarization", model = model, tokenizer = tokenizer)
                    speak("How long do you want the summary to be")
                    word_length = listen()
                    while True:
                        if (word_length is None):
                            continue
                        number_string = ''.join(re.findall(r'\d+', word_length))
                        if (number_string is None or len(number_string) == 0):
                            continue
                        else:
                            break
                    number = int(''.join(re.findall(r'\d+', word_length)))
                    words = info
                    summary = summarizer(words, max_length = max(50, number), min_length = 10, do_sample=True)[0]['summary_text']
                    summary = to_grimdark(summary)
                    speak(f"Here is a summary {summary}")
                except Exception as e:
                    speak(f"Error {e} occurred when loading summarization model")
        elif "exit" in command or "quit" in command or "stop" in command:
            speak("Goodbye")
            break
        else:
            speak("Unknown how to proceed will attempt cogitation")
            info = command
            text = f"{info}"
            response = generate_response(text, qa_pipeline, generator, gpt2_tokenizer)
            response = to_grimdark(response)
            speak(response)

if __name__ == "__main__":
    qa_pipeline, generator, gpt2_tokenizer = load_models()
    main_assistant(qa_pipeline, generator, gpt2_tokenizer)