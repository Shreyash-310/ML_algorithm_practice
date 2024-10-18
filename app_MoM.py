import warnings
warnings.filterwarnings("ignore")

import os
import whisper
import google.generativeai as genai

class GenerateTranscript:
    def __init__(self):
        self.model = whisper.load_model("medium")
        self.audio_path = "C:/Work/MoM_scipts/audio_files/"
        self.transcript_path = "C:/Work/MoM_scipts/audio_transcripts/"
        
    def get_transcribe(self, audio_file):
        # print(f"audio_file_name {audio_file}")
        trancribe_file_path = self.transcript_path+audio_file.split('.')[0]+'.txt'
        # print(trancribe_file_path)
        if os.path.exists(trancribe_file_path):
            with open(trancribe_file_path,'r') as file:
                transcript = file.readlines()
            transcript = ''.join(transcript)
        else:
            transcript = self.model.transcribe(audio_file)['text']
        return transcript

class GenerateMinutes:
    def __init__(self):
        genai.configure(api_key="AIzaSyA_bEYm4EvvktM9Xcnjv3tZnE9QJB_rjtE")
        self.model = genai.GenerativeModel('gemini-pro')
        self.save_path = "C:/Work/MoM_scipts/MoMs"
        
    def summarize(self, transcript):
        response = self.model.generate_content(["Generate the minutes of Meeting based on the following transcript", transcript])
        return response.text
    
    def save_MoM(self, mom_text):
        pass

if __name__=='__main__':
    transcript_generator = GenerateTranscript()
    minutes_generator = GenerateMinutes()
    audio_file = "audio_file_1.m4a"
    transcript = transcript_generator.get_transcribe(audio_file)
    # print(f"transcript \n{transcript}")
    mom = minutes_generator.summarize(transcript)
    print(f"Minutes of Meeting \n{mom}")