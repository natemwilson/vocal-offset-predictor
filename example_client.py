import io, os, sys, pprint, glob
from pathlib import Path

path = sys.argv[1]

# Imports the Google Cloud client library
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types

# Instantiates a client
client = speech.SpeechClient()

# The name of the audio file to transcribe
if path.endswith('.flac'):
	paths = [path]
else:
	paths = list(Path(path).rglob("*.flac"))

print(f"About to process {len(paths)} file(s).")
pprint.pprint(paths)
print(f"Press 'y' to confirm")

if input() != 'y':
	sys.exit(1)

for filename in paths:
	try:
		# Loads the audio into memory
		with io.open(filename, 'rb') as audio_file:
		    content = audio_file.read()
		    audio = types.RecognitionAudio(content=content)

		config = types.RecognitionConfig(
		    encoding=enums.RecognitionConfig.AudioEncoding.FLAC,
		    sample_rate_hertz=16000,
		    language_code='en-US',
		    enable_word_time_offsets=True)

		# Detects speech in the audio file
		response = client.recognize(config, audio)


		for w in response.results[0].alternatives[0].words:
			word = w.word
			try:
				start_time_nanos = w.start_time.nanos
			except AttributeError:
				start_time_nanos = 0

			try:
				start_time_seconds = w.start_time.seconds
			except AttributeError:
				start_time_seconds = 0

			try:
				end_time_nanos = w.end_time.nanos
			except AttributeError:
				end_time_nanos = 0

			try:
				end_time_seconds = w.end_time.seconds
			except AttributeError:
				end_time_seconds = 0


			start_time = start_time_seconds + (start_time_nanos / 10e9)
			end_time = end_time_seconds + (end_time_nanos / 10e9)
			offset = end_time - start_time

			print(f"{word}, {start_time}, {end_time}, {round(offset, 2)}")

	except Exception as e:
		print(e)

#for result in response.results:
	#print(dir(result))
    #print('Transcript: {}'.format(result.alternatives[0].transcript))
    #print(result.words.word, result.words.start_time, result.words.end_time, result.words.end_time - result.words.start_time)

