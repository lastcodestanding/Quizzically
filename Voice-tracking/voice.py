import pyaudio
import wave
import time
import speech_recognition as voicetotext
import threading
import os as operatingSystem

startingPyAudio = pyaudio.PyAudio()
audioFormat = pyaudio.paInt16
frames_per_buffer = 1024
captureRate = 44100
number_of_channels = 2

def read_audio(audioStream, nameOfFile):
    duration = 10 #seconds
    allf = list()
    for i in range(0, int(captureRate/frames_per_buffer*duration)):
        allf.append(audioStream.read(frames_per_buffer))
    waveForm = wave.open(nameOfFile, 'wb')
    waveForm.setnchannels(number_of_channels)
    waveForm.setsampwidth(startingPyAudio.get_sample_size(audioFormat))
    waveForm.setframerate(captureRate)
    waveForm.writeframes(b''.join(allf))
    waveForm.close()
    audioStream.stop_stream()
    audioStream.close()

def convert(i):
    if i >= 0:
        sound = 'record' + str(i) +'.wav'
        myRecognizer = voicetotext.Recognizer()
        with voicetotext.AudioFile(sound) as source:
            myRecognizer.adjust_for_ambient_noise(source)
            print("Performing conversion...\n")
            audio = myRecognizer.listen(source)
        try:
            value = myRecognizer.recognize_google(audio)
            operatingSystem.remove(sound)
            if str is bytes:
                result = u"{}".format(value).encode("utf-8")
            else:
                result = str(value)
            with open("transcript.txt","a") as f:
                f.write(result)
                f.write(" ")
                f.close()
        except voicetotext.UnknownValueError:
            print("")
        except voicetotext.RequestError as currentError:
            print(str(currentError))
        except KeyboardInterrupt:
            pass



def save_audio(i):
    audioStream = startingPyAudio.open(format=audioFormat,channels=number_of_channels,rate=captureRate,frames_per_buffer=frames_per_buffer,input=True)
    nameOfFile = 'record'+str(i)+'.wav'
    read_audio(audioStream, nameOfFile)

#Run for 120 seconds - 10 second recordings.
for i in range(30//10):
    thread1 = threading.Thread(target=save_audio,args=[i])
    thread2 = threading.Thread(target=convert,args=[i-1])
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()
    if i==2:
        flag = True
if flag:
    convert(i)
    startingPyAudio.terminate()
print("Done\n")
