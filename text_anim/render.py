# imports
import pandas as pd
import gizeh
from moviepy.editor import VideoClip
from math import floor
import sys

alpha = float(sys.argv[1])



# wrangle data
final_predictions = pd.read_csv('final_predictions.csv')
cols = ['word', 'word_id', 'sentence_id', 'sentence_length', 'label', 'prediction']
final_predictions = final_predictions[cols]

# select sentences
# s1
sentence1 = final_predictions[final_predictions.sentence_id == 2196]

sentence1['pred_cumsum'] = sentence1.prediction.cumsum()  * alpha
s1_pred_time = sentence1.pred_cumsum.iloc[-1] + sentence1.prediction.iloc[-1]

sentence1['label_cumsum'] = sentence1.label.cumsum() * alpha
s1_label_time = sentence1.label_cumsum.iloc[-1] + sentence1.label.iloc[-1]

sentence1.label_cumsum = sentence1.label_cumsum * (s1_pred_time/s1_label_time)

sentence1 = sentence1.reset_index()
print(sentence1[['word', 'pred_cumsum', 'label_cumsum']])

# s2
sentence2 = final_predictions[final_predictions.sentence_id == 2201]

sentence2['pred_cumsum'] = sentence2.prediction.cumsum() * alpha
s2_pred_time = sentence2.pred_cumsum.iloc[-1] + sentence2.prediction.iloc[-1]

sentence2['label_cumsum'] = sentence2.label.cumsum() * alpha
s2_label_time = sentence2.label_cumsum.iloc[-1] + sentence2.label.iloc[-1]

sentence2.label_cumsum = sentence2.label_cumsum * (s2_pred_time/s2_label_time)

sentence2 = sentence2.reset_index()
print(sentence2[['word', 'pred_cumsum', 'label_cumsum']])


# uniform sentence
sentence_u = final_predictions[final_predictions.sentence_id == 2401]

sentence_u['pred_cumsum'] = sentence_u.prediction.cumsum() * alpha
su_pred_time = sentence_u.pred_cumsum.iloc[-1] + sentence_u.prediction.iloc[-1]

sentence_u['label_cumsum'] = sentence_u.label.cumsum() * alpha
su_label_time = sentence_u.label_cumsum.iloc[-1] + sentence_u.label.iloc[-1]

sentence_u.label_cumsum = sentence_u.label_cumsum * (su_pred_time/su_label_time)

sentence_u = sentence_u.reset_index()
print(sentence_u[['word', 'pred_cumsum', 'label_cumsum']])



# set other params
h = 800
w = 1280
fps = 60

# frame-making fns

# ex1: pred 1
def make_frame_1_p(t):
    sentence = sentence1
    surface = gizeh.Surface(width=w, height=h, bg_color=(1,1,1)) # dimensions in pixel
    
    try:
        word = str(sentence[sentence.pred_cumsum > t].word.iloc[0])
    except:
        word = ''
    
    text = gizeh.text(word , fontfamily="Helvetica",  fontsize=140, fill=(0,0,0), xy=(w/2, h/2))
    text.draw(surface)
    return surface.get_npimage()

# ex2: real 1
def make_frame_1_r(t):
    sentence = sentence1
    surface = gizeh.Surface(width=w, height=h, bg_color=(1,1,1)) # dimensions in pixel
    
    try:
        word = str(sentence[sentence.label_cumsum > t].word.iloc[0])
    except:
        word = ''
    
    text = gizeh.text(word , fontfamily="Helvetica",  fontsize=140, fill=(0,0,0), xy=(w/2, h/2))
    text.draw(surface)
    return surface.get_npimage()


# ex1: pred 1
def make_frame_2_p(t):
    sentence = sentence2
    surface = gizeh.Surface(width=w, height=h, bg_color=(1,1,1)) # dimensions in pixel
    
    try:
        word = str(sentence[sentence.pred_cumsum > t].word.iloc[0])
    except:
        word = ''
    
    text = gizeh.text(word , fontfamily="Helvetica",  fontsize=140, fill=(0,0,0), xy=(w/2, h/2))
    text.draw(surface)
    return surface.get_npimage()

# ex2: real 1
def make_frame_2_r(t):
    sentence = sentence2
    surface = gizeh.Surface(width=w, height=h, bg_color=(1,1,1)) # dimensions in pixel
    
    try:
        word = str(sentence[sentence.label_cumsum > t].word.iloc[0])
    except:
        word = ''
    
    text = gizeh.text(word , fontfamily="Helvetica",  fontsize=140, fill=(0,0,0), xy=(w/2, h/2))
    text.draw(surface)
    return surface.get_npimage()


# ex3: uniform
def make_frame_u(t):
    sentence = sentence_u
    surface = gizeh.Surface(width=w, height=h, bg_color=(1,1,1)) # dimensions in pixel
    
    
    avg_wps = (sentence.index.max() + 1) / su_pred_time

    i = floor(t * avg_wps)

    try:
        word = str(sentence.word.iloc[i])
    except:
        word = ''

    
    text = gizeh.text(word , fontfamily="Helvetica",  fontsize=140, fill=(0,0,0), xy=(w/2, h/2))
    text.draw(surface)
    return surface.get_npimage()


# ex4: non-uniform
def make_frame_u_n(t):
    sentence = sentence_u
    surface = gizeh.Surface(width=w, height=h, bg_color=(1,1,1)) # dimensions in pixel
    
    try:
        word = str(sentence[sentence.label_cumsum > t].word.iloc[0])
    except:
        word = ''
    
    text = gizeh.text(word , fontfamily="Helvetica",  fontsize=140, fill=(0,0,0), xy=(w/2, h/2), v_align='center')
    text.draw(surface)
    return surface.get_npimage()

# # rendering
# clip1 = VideoClip(make_frame_1_p, duration=s1_pred_time + 3) # 3-second clip
# clip1.write_videofile(f"s1_p_{alpha}.mp4", fps=fps, preset='veryslow', audio=False) # export as video

# clip2 = VideoClip(make_frame_1_r, duration=s1_pred_time + 3) # 3-second clip
# clip2.write_videofile(f"s1_r_{alpha}.mp4", fps=fps, preset='veryslow', audio=False) # export as video

# # rendering
# clip3 = VideoClip(make_frame_2_p, duration=s2_pred_time + 3) # 3-second clip
# clip3.write_videofile(f"s2_p_{alpha}.mp4", fps=fps, preset='veryslow', audio=False) # export as video

# clip4 = VideoClip(make_frame_2_r, duration=s2_pred_time + 3) # 3-second clip
# clip4.write_videofile(f"s2_r_{alpha}.mp4", fps=fps, preset='veryslow', audio=False) # export as video

clip5 = VideoClip(make_frame_u, duration=su_pred_time + 3) # 3-second clip
clip5.write_videofile(f"su_u_{alpha}.mp4", fps=fps, preset='veryslow', audio=False) # export as video

clip6 = VideoClip(make_frame_u_n, duration=su_pred_time + 3) # 3-second clip
clip6.write_videofile(f"su_n_{alpha}.mp4", fps=fps, preset='veryslow', audio=False) # export as video
