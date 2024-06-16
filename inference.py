import torch
from transformer_implementation import transformer
from transformers import PreTrainedTokenizerFast
import numpy as np
import torch.nn.functional as F

np.random.seed(46)
torch.manual_seed(46)
# change dropout to 0
conf = {"hidden_dim": 512, "vocabulary_size": 12000, "num_heads": 8, "intermediate_dim": 2048, "eps": 1e-06, "num_layers": 6, "dropout": 0.0,
            "label_smoothing": 0.1, "warmup_steps": 4000, "max_length": 128}
device = "cuda"

tokenizer = PreTrainedTokenizerFast(tokenizer_file=r"C:\Users\Victor\Deep Learning\papers_implementation\transformers\data\tokenizer.json")
tokenizer.pad_token = "[PAD]"
tokenizer.eos_token = "[PAD]"
conf["vocabulary_size"] = tokenizer.vocab_size + 1
model = transformer(conf).to(device)
net_weights = torch.load(r'weights\transformer_200001.pth',
                         map_location={'cuda:0': 'cpu'})

model.load_state_dict(net_weights)
model.eval()
print('ネットワーク設定完了：学習済みの重みをロードしました')

text = "However, we in Parliament also have a supervisory role with regard to the Commission and we do not have to agree with everything which comes out of the Commission."
target = "Aber wir als Parlament sind auch der Kontrolleur der Kommission. Und nicht alles, was von der Kommission kommt, muß unsere Meinung sein."
inputs = tokenizer(text, add_special_tokens=True, return_tensors="pt")
print(inputs)
inputs = {k: v.to(device) for k, v in inputs.items()}
target_tokens = tokenizer(target, add_special_tokens=True, return_tensors="pt")
print(target_tokens["input_ids"][0][0])
# with torch.no_grad():
#     outputs_tokens, outputs_array = model.generate(inputs["input_ids"], tgt_max_length=32, src_attention_mask=inputs["attention_mask"], first_token=target_tokens["input_ids"][0][0], debug=True)
# print(outputs_tokens)
# print(tokenizer.decode(outputs_tokens[0], skip_special_tokens=True))

# diffs = []
# for i in range(len(outputs_array)):
#     gap = 0
#     for j in range(i, len(outputs_array)):
#         gap += abs(outputs_array[i] - outputs_array[j])
#     diffs.append(gap)
    
# print(f"Difference between outputs {diffs}")s

with torch.no_grad():
    target_tokens = {k: v.to(device) for k, v in target_tokens.items()}

    logits, loss, last_hidden_states, attentions_array = model(inputs["input_ids"], target_tokens["input_ids"], inputs["attention_mask"], target_tokens["attention_mask"], last_hidden_states=True, output_attentions=True)
    print(loss)
    # print(logits.shape)
    # for i in range(logits.shape[1]):
    #     next_token_logits = logits[:, i, :]
    #     topk_scores = torch.topk(next_token_logits, dim=-1, k=20)
    #     indices_to_remove = next_token_logits < topk_scores[0][..., -1, None]
    #     scores_processed = next_token_logits.masked_fill(indices_to_remove, torch.finfo(next_token_logits.dtype).min)
    #     scores = F.softmax(scores_processed, dim=-1)
    #     print(f"Scores of the token {i} : {torch.topk(scores, dim=-1, k=20)}")
    diffs = []
    for i in range(len(last_hidden_states)):
        gap = 0
        for j in range(len(last_hidden_states)):
            if j == i:
                gap += abs(last_hidden_states[i] - last_hidden_states[j])
        diffs.append(gap)
        
    print(f"Difference between each hidden state: {diffs}")
    print("*"*50)
    diffs = []
    for i in range(len(attentions_array)):
        gap = 0
        for j in range(len(attentions_array)):
            if j == i:
                gap += abs(attentions_array[i] - attentions_array[j])
        diffs.append(torch.sum(gap))
        
    print(f"Difference between each attention layer: {diffs}")
    print(attentions_array[0][0].shape)