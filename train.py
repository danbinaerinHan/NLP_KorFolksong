import os 

import torch 
import torch.nn 
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import pickle

from transformers import WhisperProcessor, WhisperTokenizer, WhisperForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import nlptutti as metrics
import torch.optim as optim
from tqdm.auto import tqdm 
import wandb
import argparse
import datetime
import time

import whisper
#from my_model import Mymodel
from data_utils import MinyoDataset, custom_collate_fn, get_wer
from my_model_daewoung import Mymodel

def get_argument_parser():
  parser = argparse.ArgumentParser()

  parser.add_argument('--batch_size', type=int, default=8)
  parser.add_argument('--lr', type=float, default=0.0001)
  parser.add_argument('--num_epochs', type=int, default=15)
  parser.add_argument('--max_len', type=int, default=1024)
  parser.add_argument('--n_ref_encoder_layer', type=int, default=3)
  parser.add_argument('--n_ref_decoder_layer', type=int, default=3)
  parser.add_argument('--n_ref_text_ctx', type=int, default=1024)
  parser.add_argument('--n_ref_text_state', type=int, default=1280)
  
  parser.add_argument('--random_ratio', type=float, default=0.1)

  parser.add_argument('--num_workers', type=int, default=8)
  parser.add_argument('--device', type=str, default='cuda')
  return parser

def make_experiment_name_with_date(args):
  current_time_in_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  return f'{current_time_in_str}-{args.n_ref_encoder_layer}_{args.n_ref_decoder_layer}_{args.n_ref_decoder_layer}_{args.n_ref_text_state}'



def main():
  args = get_argument_parser().parse_args()
  wandb.init(
    project="whisper-korean_folksong",  
    name = make_experiment_name_with_date(args), 
    config = args
  )
  audio_path = '/home/daewoong/userdata/danbi/final_tts_audio'
  lyric_path = '/home/daewoong/userdata/danbi/final_lyrics_data/'
  each_lyric = '/home/daewoong/userdata/danbi/each_song_lyrics.txt'
  result_path = '/home/daewoong/userdata/danbi/encoder_result'
  filtered_id_list = pickle.load(open('/home/daewoong/userdata/danbi/thirty_second_filtered_id.pkl', 'rb'))
  
  print('download token now')

  processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2", language="ko", task="transcribe", predict_timestamps=True)
  dataset = MinyoDataset(result_path, lyric_path, processor, filtered_id_list, each_lyric, max_len = args.max_len, random_ratio=args.random_ratio)
  
  print('token download complete')

  train_size = int(len(dataset) * 0.8)
  valid_size = len(dataset) - train_size

  train_data, valid_data = random_split(dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(42))

  train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=8, pin_memory=True)
  valid_dataloader = DataLoader(valid_data, batch_size=16, shuffle=False, collate_fn=custom_collate_fn, num_workers=8, pin_memory=True)

  #next(iter(train_dataloader))


  print('load model now')
  #pre_model = whisper.load_model("large-v2")
  pre_model = whisper.load_model("/home/daewoong/userdata/danbi/whisper_pretrain/large-v2.pt", device='cpu')  
  #WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
  print('load model complete')

  processor.tokenizer.set_prefix_tokens(language="ko", task="transcribe", predict_timestamps=True)
  # pre_model.config.forced_decoder_ids = None
  # pre_model.config.suppress_tokens = []



  criterion = nn.CrossEntropyLoss(ignore_index = -100)

  device = 'cuda'
  epoch = args.num_epochs
  model_dims = pre_model.dims
  model_dims.n_ref_encoder_layer = args.n_ref_encoder_layer
  model_dims.n_ref_decoder_layer = args.n_ref_decoder_layer
  model_dims.n_ref_text_ctx = args.n_ref_text_ctx
  model_dims.n_ref_text_state = args.n_ref_text_state

  model = Mymodel(model_dims)

  model.encoder.load_state_dict(pre_model.encoder.state_dict())
  model.decoder.token_embedding.load_state_dict(pre_model.decoder.token_embedding.state_dict())
  model.decoder.blocks.load_state_dict(pre_model.decoder.blocks.state_dict())
  model.decoder.positional_embedding.data=pre_model.decoder.positional_embedding.data.clone()
  model.ref_encoder.token_embedding.load_state_dict(pre_model.decoder.token_embedding.state_dict())
  

    
  model = model.to(device)
  
  for param in model.encoder.parameters():
    param.requires_grad = False
  for param in model.decoder.blocks.parameters():
    param.requires_grad = False
  
  optimizer = optim.AdamW(model.parameters(), lr=args.lr)

  train_loss_record = []
  train_wer_record= []
  train_cer_record= []
  train_crr_record= []

  valid_loss_record = []
  valid_wer_record= []
  valid_cer_record= []
  valid_crr_record= []

  best_val_wer = 1.0
  
  for i in tqdm(range(epoch)):
    
    batch_wer_n = 0
    batch_cer_n = 0
    batch_wer_s = 0
    batch_wer_d = 0 
    batch_wer_i = 0 
    batch_cer_s = 0 
    batch_cer_d = 0
    batch_cer_i = 0
    
    val_batch_wer_n = 0
    val_batch_cer_n = 0
    val_batch_wer_s = 0
    val_batch_wer_d = 0
    val_batch_wer_i = 0
    val_batch_cer_s = 0
    val_batch_cer_d = 0
    val_batch_cer_i = 0

    
    # model.train()
    
    
    train_loss_record = []
    train_wer_record= []
    train_cer_record= []

    valid_loss_record = []
    valid_wer_record= []
    valid_cer_record= []
      
    model.train()
    
    for idx, batch in (enumerate(tqdm(train_dataloader, leave=False))):
      audio, input_text, input_txt_attn, around_text, around_txt_attn = batch
      x_input_text = input_text[:,:-1] #[batch, seq_len-1]
      # train_batch += input_text.size(0)
      true_input_text = input_text[:,1:] #[batch, seq_len-1]
      
      pred = model(audio.to(device), around_text.to(device), tokens=x_input_text.to(device))
      
      true_input_text_with_mask = true_input_text.masked_fill(input_txt_attn[:, 1:].ne(1) , -100)
      
      #pred [batch, seq_len, ]
      loss = criterion(pred.view(-1, pred.size(-1)), true_input_text_with_mask.view(-1).to(device))
      wandb.log({"train_loss": loss.item()})
      # wers_n, wers_s, wers_d, wers_i, cers_n, cers_s, cers_d, cers_i = get_wer(pred, true_input_text, processor)
      
      # batch_wer_n += wers_n
      # batch_wer_s += wers_s
      # batch_wer_d += wers_d
      # batch_wer_i += wers_i
      # batch_cer_n += cers_n
      # batch_cer_s += cers_s
      # batch_cer_d += cers_d
      # batch_cer_i += cers_i
      
      train_loss_record.append(loss.item())
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      
      # train_wer += wers
      # train_crr += crrs
      # train_cer += cers
      
    #   if idx % 50 == 0:
    #     print("loss : ", loss.item(), "wer : ", (batch_wer_s + batch_wer_d + batch_wer_i) / batch_wer_n , "cer : ", (batch_cer_s + batch_cer_d + batch_cer_s) / batch_cer_n)
      
    
    # train_wer_record.append((batch_wer_s + batch_wer_d + batch_wer_i) / batch_wer_n)
    # train_cer_record.append((batch_cer_s + batch_cer_d + batch_cer_s) / batch_cer_n)

    # print("wer : ", train_wer / train_batch, "cer : ", train_cer / train_batch, "crr : ", train_crr / train_batch)
    # torch.save(model.state_dict(), f'/home/daewoong/userdata/first_train/minyo_decoder_layer_{i}.pth')

    model.eval()
    
    with torch.inference_mode():
      valid_loss = 0
      for batch in tqdm(valid_dataloader, leave=False):
        audio, input_text, input_txt_attn, around_text, around_txt_attn = batch
        x_input_text = input_text[:,:-1] #[batch, seq_len-1]
        # train_batch += input_text.size(0)
        true_input_text = input_text[:,1:] #[batch, seq_len-1]
        
        pred = model(audio.to(device), around_text.to(device), tokens=x_input_text.to(device))
        
        true_input_text_with_mask = true_input_text.masked_fill(input_txt_attn[:, 1:].ne(1) , -100)
        
        #pred [batch, seq_len, ]
        loss = criterion(pred.view(-1, pred.size(-1)), true_input_text_with_mask.view(-1).to(device))
        
        wers_n, wers_s, wers_d, wers_i, cers_n, cers_s, cers_d, cers_i = get_wer(pred, true_input_text, processor)
        
        val_batch_wer_n += wers_n
        val_batch_wer_s += wers_s
        val_batch_wer_d += wers_d
        val_batch_wer_i += wers_i
        val_batch_cer_n += cers_n
        val_batch_cer_s += cers_s
        val_batch_cer_d += cers_d
        val_batch_cer_i += cers_i
        
        valid_loss_record.append(loss.item())
        valid_loss += loss.item() * valid_dataloader.batch_size
        if idx % 50 == 0:
          print("valid wer : ", (val_batch_wer_s + val_batch_wer_d + val_batch_wer_i) / val_batch_wer_n , "valid cer : ", (val_batch_cer_s + val_batch_cer_d + val_batch_cer_s) / val_batch_cer_n)
        
        # valid_wer += wers
        # valid_crr += crrs
        # valid_cer += cers
        # print(f'wer : {wers / input_text.size(0)}, cers:{cers / input_text.size(0)}, crrs:{crrs / input_text.size(0)}')
      total_valid_loss = valid_loss / len(valid_data)
      wandb.log({"valid_loss": total_valid_loss})
      valid_wer_record.append((val_batch_wer_s + val_batch_wer_d + val_batch_wer_i) / val_batch_wer_n)
      valid_cer_record.append((val_batch_cer_s + val_batch_cer_d + val_batch_cer_s) / val_batch_cer_n)    
      # valid_wer_record.append(valid_wer / valid_batch)
      # valid_cer_record.append(valid_cer / valid_batch)
      wandb.log({"valid_wer": (val_batch_wer_s + val_batch_wer_d + val_batch_wer_i) / val_batch_wer_n})
      wandb.log({"valid_cer": (val_batch_cer_s + val_batch_cer_d + val_batch_cer_s) / val_batch_cer_n})
      if ((val_batch_wer_s + val_batch_wer_d + val_batch_wer_i) / val_batch_wer_n) < best_val_wer:
        best_val_wer = (val_batch_wer_s + val_batch_wer_d + val_batch_wer_i) / val_batch_wer_n
        if i % 5 == 0:
          torch.save({'model':self.model.state_dict(), 'optim':self.optimizer.state_dict()}, f'//home/daewoong/userdata/danbi/train_result/val_best_wer_{i}.pt')
          print("save model")
        
      print("valid final val_wer : ", (batch_wer_s + batch_wer_d + batch_wer_i) / (batch_wer_n+1e-5), "val_cer : ",(batch_cer_s + batch_cer_d + batch_cer_s) / (batch_cer_n+1e-5))
      if i % 5 == 0:
        torch.save({'model':self.model.state_dict(), 'optim':self.optimizer.state_dict()}, f'/home/daewoong/userdata/danbi/train_result/epoch_{i}.pt') # last epoch model save

    
    # plt.figure(figsize=(20, 10))
    # plt.subplot(2, 3, 1)
    # plt.plot(train_loss_record)
    # plt.title("train_loss")
    # plt.subplot(2, 3, 2)
    # plt.plot(train_wer_record)
    # plt.title("train_wer")
    # plt.subplot(2, 3, 3)
    # plt.plot(train_cer_record)
    # plt.title("train_cer")
    # plt.subplot(2, 3, 4)
    # plt.plot(valid_loss_record)
    # plt.title("valid_loss")
    # plt.subplot(2, 3, 5)
    # plt.plot(valid_wer_record)
    # plt.title("valid_wer")
    # plt.subplot(2, 3, 6)
    # plt.plot(valid_cer_record)
    # plt.title("valid_cer")
    # plt.savefig('/home/daewoong/userdata/danbi/train_result/layers3_train.png')
    
  wandb.finish()  
if __name__ == '__main__':
  main()