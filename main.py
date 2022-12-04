import pyttsx3
import datetime
import speech_recognition as sr
import webbrowser as wb
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tflearn
import tensorflow
import random
import json

rosta=pyttsx3.init()
voice=rosta.getProperty('voices')
rosta.setProperty('voice',voice[1].id)

def speak(audio):
    print("Rosta: " + audio)
    rosta.say(audio)
    rosta.runAndWait()
def time():
    Time=datetime.datetime.now().strftime("%I:%M:%p")
    speak(Time)

def welcome():
    hour=datetime.datetime.now().hour
    if hour >=5 and hour <12:
        speak("Good morning")
    elif hour >=12 and hour <18:
        speak("Good afternoon")
    elif hour >=18 and hour <=24:
        speak("Good night")
    elif hour <5:
        speak("Good night")
    speak("How can I help you?")

def command():
    c=sr.Recognizer()
    with sr.Microphone() as source:
        c.adjust_for_ambient_noise(source,duration=1)
        audio=c.listen(source)
    try:
        query=c.recognize_google(audio, language='vi-VN')
        print("Duy Anh: " + query)
    except sr.UnknownValueError:
        print("Please repeat or typing your command")
        query=str(input('Your order is: '))
    return query


if __name__ == "__main__":
    welcome()
    print(command())
    while True:
        query=command().lower()
        if "google" in query:
            speak("What should I search?")
            search=command().lower()
            url=f"https://www.google.com/search?q={search}"
            wb.get().open(url)
            speak(f'Here is your {search} on google')




with open("intents.json") as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(pattern)
        docs_y.append(intent["tag"])
    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]
for x, doc in enumerate(docs_x):
    bag = []
    wrds = [stemmer.stem(w) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1
    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")