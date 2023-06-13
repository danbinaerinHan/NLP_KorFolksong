from pathlib import Path
import re

from pathlib import Path
import random
import torchaudio

class TextProcessor():
  def __init__(self, id_list, lyric_path, each_lyric, random_ratio=0.8):
    self.lyric_path = Path(lyric_path)
    self.filtered_id = id_list
    self.lyrics_list = [lyric_path+i+'.txt' for i in self.filtered_id]
    self.each_lyric_list = self.read_text(each_lyric).split('\n')
    self.random_ratio = random_ratio
    # self.teach_forcing_prob = teach_forcing_prob
  
  def read_text(self, txt_path):
    with open(txt_path, 'r') as f:
      text = f.read()
    return text
  
  def get_all_sentence_list(self):
    all_sentence = []
    for i in self.lyrics_list:
      all_sentence.append(self.read_text(i))
    return list(set(all_sentence))
  
  def __len__(self):
    return len(self.lyrics_list)
  
  def get_random_sentence(self, lyric_name, random_mode=False):
    # lyric_name = 'lyric0-1.txt'
    if random.random() < self.random_ratio:
      random_mode = True
    if random_mode == False:
      lyric_index = int(lyric_name.split('-')[0][5:])
      lyrics = self.each_lyric_list[lyric_index]
      lyrics = lyrics.split(', ')
      if len(lyrics) > 30:
        lyrics = random.sample(lyrics, 30)
      # print(lyrics)
      random.shuffle(lyrics)
    else: 
      lyrics = random.sample(self.get_all_sentence_list(), 30)
      
    return lyrics
  
  def __getitem__(self, idx):
    txt_path = self.lyrics_list[idx]
    lyric_name = txt_path.split('/')[-1]
    # print(lyric_name)
    input_text = self.read_text(txt_path)
    random.seed(0)
    output_sentence = self.get_random_sentence(lyric_name)
    return input_text, output_sentence

# lyric_path = '/home/daewoong/userdata/danbi/final_lyrics_data'
# each_lyric = '/home/daewoong/userdata/danbi/each_song_lyrics.txt'
# processor = TextProcessor(lyric_path, each_lyric)
  
# sorted 된거 불러오기.

#audio_pth=Path('/home/daewoong/userdata/latest_tts'), text_pth = Path('/home/daewoong/userdata/latest_txt')


# #(audio_pth=Path('/home/daewoong/userdata/real_latest_tts'), text_pth = Path('/home/daewoong/userdata/real_latest_txt'))
# def get_sorted_pth(audio_pth=Path('/home/daewoong/userdata/latest_tts'), text_pth = Path('/home/daewoong/userdata/latest_txt')):
#   audio_pth_list = [] 
  
#   for pth in audio_pth.glob('*.wav'):
#     audio_pth_list.append(pth)
#   audio_pth_list.sort()

#   txt_pth_list = [] 
#   for pth in text_pth.glob('*.txt'):
#     txt_pth_list.append(pth)
#   txt_pth_list.sort()
#   sorted_audio_paths = sorted(audio_pth_list, key=get_number_from_path)    
#   sorted_txt_paths = sorted(txt_pth_list, key=get_number_from_path)  
  
#   return sorted_audio_paths, sorted_txt_paths


# 30초 이상은 잘린거 불러오기
# def get_filtered_pth():
#   sorted_audio_paths, sorted_txt_paths = get_sorted_pth()
#   filtered_audio_paths = [i for idx,i in enumerate(sorted_audio_paths) if idx not in exclude_list]
#   filtered_txt_paths = [i for idx,i in enumerate(sorted_txt_paths) if idx not in exclude_list]
#   assert len(filtered_audio_paths) == len(filtered_txt_paths)
#   assert len(filtered_audio_paths) == len(sorted_audio_paths) - len(exclude_list)
#   assert len(filtered_txt_paths) == len(sorted_txt_paths) - len(exclude_list)
#   return filtered_audio_paths, filtered_txt_paths


# def get_each_lyric(sorted_txt= '/home/daewoong/userdata/each_lyrics.txt'):

#   with open(sorted_txt, 'r') as f:
#     sorted_list = f.readlines()

#   each_lyric = {}
#   skip_next_first = False 
#   final_len = 0

#   for song_idx, i in enumerate(sorted_list):
#     p = [txt.strip() for txt in i.split(',')]
    
#     if skip_next_first:
#       final_len += 1
#       p = p[1:]
#       skip_next_first = False 
    
#     if len(p) % 2 == 1:
#       len_p = len(p) - 1
#     else : 
#       len_p = len(p)
      
#     for txt_len in range(0, len_p, 2):
#         #sentence = p[txt_len] + ' ' + p[txt_len+1]
#         each_lyric[final_len + (txt_len // 2)] = song_idx
        
#     if len(p) % 2 == 1:
#       final_len += (txt_len // 2) + 1
#       each_lyric[final_len] = (song_idx, song_idx+1)
#       skip_next_first = True

#       #final_len += sentence
    
#     else:  
#       final_len += ((txt_len // 2) + 1)

#   return each_lyric
def filter_length(audio_path, max_len=30):
  audio_list = [str(i) for i in audio_path.rglob('*.wav')]
  filtered_audio_list = []
  for i in audio_list:
    audio, sr = torchaudio.load(i)
    if len(audio)/sr <= float(max_len):
      filtered_audio_list.append(i)
  return filtered_audio_list








# def get_each_lyric(sorted_txt= '/home/daewoong/userdata/each_lyrics2.txt'):
#     with open(sorted_txt, 'r') as f:
#         sorted_list = f.readlines()

#     each_lyric = {}
#     total_len = 0
#     last_line_remaining = None
#     last_song_idx = None

#     for song_idx, line in enumerate(sorted_list):
#         line_parts = [txt.strip() for txt in line.split(',')]

#         if last_line_remaining:
#             each_lyric[total_len] = (last_song_idx, song_idx)
#             total_len += 1
#             line_parts = line_parts[1:]
#             last_line_remaining = None

#         for i in range(0, len(line_parts) - 1, 2):
#             each_lyric[total_len] = song_idx
#             total_len += 1

#         if len(line_parts) % 2 == 1:
#             last_line_remaining = line_parts[-1]
#             last_song_idx = song_idx

#     if last_line_remaining:
#         each_lyric[total_len] = (last_song_idx, len(sorted_list))

#     return each_lyric




# text_pth = Path('/home/daewoong/userdata/latest_txt')
# audio_pth = Path('/home/daewoong/userdata/latest_tts')


# exclude_list = [201,
#  365,
#  484,
#  642,
#  1004,
#  1033,
#  1146,
#  1245,
#  1260,
#  1271,
#  1608,
#  1617,
#  1751,
#  1777,
#  2454,
#  2468,
#  2502,
#  2647,
#  2863,
#  2964,
#  3203,
#  3816,
#  4188,
#  4527,
#  4528,
#  4531,
#  4532,
#  4533,
#  4578,
#  4862,
#  4872,
#  5245,
#  5270,
#  5323,
#  5354,
#  5356,
#  5361,
#  5363,
#  5364,
#  5443,
#  5446,
#  5447,
#  5543,
#  5746,
#  5749,
#  5761,
#  5771,
#  6024,
#  6041,
#  6043,
#  6063,
#  6417,
#  6699,
#  6701,
#  7178,
#  7293,
#  7331,
#  7594,
#  8158,
#  8547,
#  8549,
#  9158,
#  11595,
#  12915,
#  13522,
#  14364,
#  14806,
#  15488,
#  16319,
#  17305,
#  18146,
#  18641,
#  20021,
#  20066,
#  20371,
#  20418,
#  20497,
#  20895,
#  21076,
#  21078,
#  21101,
#  21107,
#  21114,
#  21117,
#  21543,
#  21884,
#  21887,
#  22718,
#  22871,
#  22931,
#  23383,
#  23559,
#  23737,
#  23796,
#  23819,
#  23821,
#  23856,
#  23997,
#  24206,
#  24870,
#  24893,
#  24922,
#  24974,
#  24975,
#  25168,
#  25305,
#  25466,
#  25675,
#  25748,
#  25777,
#  25942,
#  25943,
#  25956,
#  25965,
#  25998,
#  26091,
#  26110,
#  26116,
#  26140,
#  26305,
#  26357,
#  26477,
#  26480,
#  26490,
#  26507,
#  26511,
#  26797,
#  26823,
#  27409,
#  27425,
#  27569,
#  27858,
#  27862,
#  27863,
#  27864,
#  27867,
#  27869,
#  28076,
#  28201,
#  28213,
#  28218,
#  28248,
#  28807,
#  29022,
#  29218,
#  29226,
#  29286,
#  29426,
#  29605,
#  29705,
#  29842,
#  29897,
#  29923,
#  30138,
#  30680,
#  30693,
#  30701,
#  30735,
#  30737,
#  30757,
#  30913,
#  31114,
#  31134,
#  31477,
#  31541,
#  31824,
#  32060,
#  32168,
#  32215,
#  32277,
#  32604,
#  32687,
#  32722,
#  32767,
#  32795,
#  32797,
#  33025,
#  33091,
#  33185,
#  33287,
#  33688,
#  33993,
#  34546,
#  34549,
#  34555,
#  36888,
#  38104,
#  38108,
#  38849,
#  39600,
#  39787,
#  40997,
#  41001,
#  41769,
#  46911,
#  47126,
#  48345,
#  48350,
#  48352,
#  48353,
#  48354,
#  48355,
#  48360,
#  48998,
#  49112,
#  49475,
#  50689,
#  50697,
#  50794,
#  51217,
#  51220,
#  54369,
#  54527,
#  54560,
#  54566,
#  55696,
#  56405,
#  56817,
#  56891,
#  58488,
#  58773,
#  59455,
#  59460,
#  59516,
#  59582,
#  59875,
#  60138,
#  60142,
#  60162,
#  60167,
#  60168,
#  60173,
#  60178,
#  60179,
#  60183,
#  60184,
#  60185,
#  60186,
#  60187,
#  60188,
#  60189,
#  60657,
#  60658,
#  60661,
#  61068,
#  61209,
#  61607,
#  61619,
#  61624,
#  61725,
#  61759,
#  61762,
#  61987,
#  62321,
#  62925,
#  62928,
#  62932,
#  62933,
#  62949,
#  63102,
#  63211,
#  63835,
#  63977,
#  64430,
#  64525,
#  64563,
#  64564,
#  64566,
#  64574,
#  64578,
#  64737,
#  64797,
#  64911,
#  65080,
#  65149,
#  65152,
#  65514,
#  65515,
#  65549,
#  65550,
#  66016,
#  66148,
#  66863,
#  66864,
#  67214,
#  67408,
#  68140,
#  68149,
#  68152,
#  68153,
#  68162,
#  68169]


exclude_list = [201,
 365,
 484,
 642,
 1004,
 1033,
 1146,
 1245,
 1260,
 1271,
 1608,
 1617,
 1751,
 1777,
 2454,
 2468,
 2502,
 2647,
 2863,
 2964,
 3203,
 3816,
 4188,
 4527,
 4528,
 4531,
 4532,
 4533,
 4578,
 4862,
 4872,
 5245,
 5270,
 5323,
 5354,
 5356,
 5361,
 5363,
 5364,
 5443,
 5446,
 5447,
 5543,
 5746,
 5749,
 5761,
 5771,
 6024,
 6041,
 6043,
 6063,
 6417,
 6699,
 6701,
 7178,
 7293,
 7331,
 7594,
 8158,
 8547,
 8549,
 9158,
 11595,
 12915,
 13522,
 14364,
 14806,
 15488,
 16319,
 17305,
 18146,
 18641,
 20021,
 20066,
 20371,
 20418,
 20497,
 20895,
 21076,
 21078,
 21101,
 21107,
 21114,
 21117,
 21543,
 21884,
 21887,
 22718,
 22871,
 22931,
 23383,
 23559,
 23737,
 23796,
 23819,
 23821,
 23856,
 23997,
 24206,
 24870,
 24893,
 24922,
 24974,
 24975,
 25168,
 25305,
 25466,
 25675,
 25748,
 25777,
 25942,
 25943,
 25956,
 25965,
 25998,
 26091,
 26110,
 26116,
 26140,
 26305,
 26357,
 26477,
 26480,
 26490,
 26507,
 26511,
 26797,
 26823,
 27409,
 27425,
 27569,
 27858,
 27862,
 27863,
 27864,
 27867,
 27869,
 28076,
 28201,
 28213,
 28218,
 28248,
 28807,
 29022,
 29218,
 29226,
 29286,
 29426,
 29605,
 29705,
 29842,
 29897,
 29923,
 30138,
 30680,
 30693,
 30701,
 30735,
 30737,
 30757,
 30913,
 31114,
 31134,
 31477,
 31541,
 31824,
 32060,
 32168,
 32215,
 32277,
 32604,
 32687,
 32722,
 32767,
 32795,
 32797,
 33025,
 33091,
 33185,
 33287,
 33688,
 33993,
 34546,
 34549,
 34555,
 36888,
 38104,
 38108,
 38849,
 39600,
 39787,
 40997,
 41001,
 41769,
 46911,
 47126,
 48345,
 48350,
 48352,
 48353,
 48354,
 48355,
 48360,
 48998,
 49112,
 49475,
 50689,
 50697,
 50794,
 51217,
 51220,
 54369,
 54527,
 54560,
 54566,
 55696,
 56405,
 56817,
 56891,
 58488,
 58773,
 59455,
 59460,
 59516,
 59582,
 59875,
 60138,
 60142,
 60162,
 60167,
 60168,
 60173,
 60178,
 60179,
 60183,
 60184,
 60185,
 60186,
 60187,
 60188,
 60189,
 60657,
 60658,
 60661,
 61068,
 61209,
 61607,
 61619,
 61624,
 61725,
 61759,
 61762,
 61987,
 62321,
 62925,
 62928,
 62932,
 62933,
 62949,
 63102,
 63211,
 63835,
 63977,
 64430,
 64525,
 64563,
 64564,
 64566,
 64574,
 64578,
 64737,
 64797,
 64911,
 65080,
 65149,
 65152,
 65514,
 65515,
 65549,
 65550,
 66016,
 66148,
 66863,
 66864,
 67214,
 67408,
 68140,
 68149,
 68152,
 68153,
 68162,
 68169]