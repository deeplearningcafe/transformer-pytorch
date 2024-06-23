import torch
import numpy as np
import torch.nn.functional as F
from util.prepare_model import prepare_test
import hydra
from omegaconf import DictConfig

np.random.seed(46)
torch.manual_seed(46)
# change dropout to 0
def generate(model, inputs:torch.tensor, bos_token:int=None, stop_tokens:list[int]=None):
    with torch.no_grad():
        outputs_tokens, outputs_array = model.generate(inputs["input_ids"], tgt_max_length=32, src_attention_mask=inputs["attention_mask"], bos_token=bos_token, stop_tokens=stop_tokens, debug=True)
    print(outputs_tokens)
    # print(tokenizer.decode(outputs_tokens[0], skip_special_tokens=True))

    diffs = []
    for i in range(len(outputs_array)):
        gap = 0
        for j in range(i, len(outputs_array)):
            gap += torch.sum(torch.abs(outputs_array[0][i] - outputs_array[0][j]))/outputs_array.shape[-1]
        diffs.append(gap)
        
    print(f"Difference between outputs {diffs}")
    return outputs_tokens, outputs_array



def debug_inference(model, inputs:torch.tensor, target_tokens:torch.tensor):
    with torch.no_grad():

        logits, loss, last_hidden_states, attentions_array, activations_array = model(inputs["input_ids"], target_tokens["input_ids"], inputs["attention_mask"], target_tokens["attention_mask"], last_hidden_states=True, output_attentions=True, output_activation_state=True)
        print(loss)
        print(logits.shape)
        for i in range(logits.shape[1]):
            next_token_logits = logits[:, i, :]
            topk_scores = torch.topk(next_token_logits, dim=-1, k=10)
            indices_to_remove = next_token_logits < topk_scores[0][..., -1, None]
            scores_processed = next_token_logits.masked_fill(indices_to_remove, torch.finfo(next_token_logits.dtype).min)
            scores = F.softmax(scores_processed, dim=-1)
            print(f"Scores of the token {i} : {torch.topk(scores, dim=-1, k=10)}")
        
        diffs = []
        for i in range(len(last_hidden_states)):
            gap = 0
            for j in range(len(last_hidden_states)):
                if j != i:
                    gap += torch.sum(torch.abs(last_hidden_states[0][i] - last_hidden_states[0][j]))/last_hidden_states.shape[-1]
            diffs.append(gap)
            
        print(f"Difference between each hidden state: {diffs}")
        print("*"*50)
        diffs = []
        for i in range(len(attentions_array)):
            gap = 0
            for j in range(len(attentions_array)):
                if j != i:
                    gap += torch.sum(torch.abs(attentions_array[i][0][0] - attentions_array[j][0][0]))/attentions_array[0][0][0].shape[-1]
            diffs.append(torch.sum(gap))
            
        print(f"Difference between each attention layer: {diffs}")
        print(attentions_array[0][0].shape)

        print(activations_array[0])


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(conf: DictConfig):

    model, tokenizer = prepare_test(conf)
    print('ネットワーク設定完了：学習済みの重みをロードしました')

    text = "However, we in Parliament also have a supervisory role with regard to the Commission and we do not have to agree with everything which comes out of the Commission."
    target = "Aber wir als Parlament sind auch der Kontrolleur der Kommission. Und nicht alles, was von der Kommission kommt, muß unsere Meinung sein."
    inputs = tokenizer(text, add_special_tokens=True, return_tensors="pt", return_token_type_ids=False)
    print(inputs)
    inputs = {k: v.to(conf.train.device) for k, v in inputs.items()}
    target_tokens = tokenizer(target, add_special_tokens=True, return_tensors="pt", return_token_type_ids=False)
    target_tokens = {k: v.to(conf.train.device) for k, v in target_tokens.items()}
    print(target_tokens["input_ids"][0][0])

    stop_tokens = [tokenizer.eos_token_id, tokenizer.pad_token_id]
    outputs_tokens, outputs_array = generate(model, inputs, bos_token=tokenizer.bos_token_id, stop_tokens=stop_tokens)
    print(tokenizer.decode(outputs_tokens[0], skip_special_tokens=False))

    print("*"*50)
    debug_inference(model, inputs, target_tokens)

if __name__ == "__main__":
    main()