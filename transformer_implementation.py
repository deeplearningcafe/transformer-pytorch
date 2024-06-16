import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

# parts: tokenizer, input embedding, positional encoding, multi head self attention, feed forward network, layer norm


class Tokenizer:
    def __init__(self) -> None:
        pass

class embeddings(nn.Module):
    def __init__(self, conf) -> None:
        super(embeddings, self).__init__()

        self.hidden_dim = conf["hidden_dim"]
        self.vocabulary_size = conf["vocabulary_size"]

        # self.weights = nn.Parameter(torch.empty((self.vocabulary_size, self.hidden_dim), dtype=torch.float32))
        self.weights = nn.Embedding(self.vocabulary_size, self.hidden_dim, padding_idx=3)

    def forward(self, x:torch.tensor):
        """Forward method of the embeddings class, given a tensor with the ids, it returns the embeddings of each token 

        Args:
            x (torch.tensor): tensor containing the tokens ids
        """
        x = self.weights(x) * torch.sqrt(torch.tensor(self.hidden_dim))
        
        return x
            

# not learned so no need to be nn.Module
class positional_encoding(nn.Module):
    def __init__(self,conf) -> None:
        super(positional_encoding, self).__init__()

        self.hidden_dim = conf["hidden_dim"]
        # self.dropout = conf["dropout"]
        self.max_seq_length = conf["max_length"]
        self.encoding = self.generate_encoding(self.max_seq_length)
        self.dropout = nn.Dropout(p=conf["dropout"])
        
    def generate_encoding(self, seq_length):
        pe = torch.zeros((seq_length, self.hidden_dim), requires_grad=False)

        # we just need to compute this once, then is just the same always
        for pos in range(seq_length):
            for i in range(0, self.hidden_dim//2):
                # incorrect implementation, 2*(i+1) is wrong
                pe[pos, 2*i] = torch.sin((pos/torch.pow(torch.tensor(10000.0), 2.0*i/self.hidden_dim)))
                pe[pos, (2*i)+1] = torch.cos((pos/torch.pow(torch.tensor(10000.0), 2.0*i/self.hidden_dim)))

        return pe
    
    def forward(self, x:torch.tensor) -> torch.tensor:
        """Forward of the positional embeddings class

        Args:
            x (torch.tensor): tensor containing the embeddings. [batch_size, seq_len, hidden_size]

        Returns:
            torch.tensor: [batch_size, seq_len, hidden_size]
        """
        # loop all the batch, no need, just sum and all ok
        # for i in range(x.shape[0]):
        # x -> [batch_size, seq_len, hidden_dim] and self.encoding -> [max_seq_len, hidden_dim]
        x += self.encoding[:x.shape[1]].to(x.device)
        
        return self.dropout(x)

    def generate(self, x:torch.tensor, idx:int):
        x += self.encoding[idx].to(x.device)
        # we remove the dropout as we are in inference mode
        return x
    
class multi_head_self_attention(nn.Module):
    def __init__(self, conf) -> None:
        super(multi_head_self_attention, self).__init__()

        self.hidden_dim = conf["hidden_dim"]
        self.num_heads = conf["num_heads"]
        self.head_dim = self.hidden_dim // self.num_heads

        self.q = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim, bias=False)
        self.k = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim, bias=False)
        self.v = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim, bias=False)

        self.output_projection = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim, bias=False)

    def forward(self, x, attention_mask=None, context=None, output_attentions:bool=False):
        batch_size, seq_length, _ = x.shape
        seq_length_key_value = seq_length
        # projection
        query = self.q(x)
        if context is not None:
            _, seq_length_key_value, _ = context.shape
            key = self.k(context)
            value = self.v(context)
        else:
            key = self.k(x)
            value = self.v(x)

        # multi head division, the order has to be [batch_size, seq_length, num_heads, head_dim] because we want to divide the hidden dim, then we just transpose
        # QxK: [batch_size, num_heads, seq_length, head_dim] X [batch_size, num_heads, head_dim, seq_length] ->  [batch_size, num_heads, seq_length, seq_length]
        # (QxK)xV: [batch_size, num_heads, seq_length, seq_length] x [batch_size, num_heads, seq_length, head_dim] -> [batch_size, num_heads, seq_length, head_dim]
        query = query.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_length_key_value, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_length_key_value, self.num_heads, self.head_dim).transpose(1, 2)

        # scaled dot product attention
        attention = torch.matmul(query, key.transpose(3, 2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        if attention_mask != None:
            # print("using mask")
            # print(key.shape, attention.shape, attention_mask.shape)
            attention += attention_mask[:, :, :query.shape[-2], :key.shape[-2]]

        attention = F.softmax(attention, dim=-1)
        # [batch_size, num_heads, seq_length, head_dim]
        attention_output = torch.matmul(attention, value)
        # [batch_size, seq_length, num_heads, head_dim] here we need the continuous() else we get an error because of the strides of the tensor
        attention_output = attention_output.transpose(1, 2).contiguous()
        # print(attention.shape)

        x = attention_output.view(batch_size, seq_length, self.hidden_dim)
        x = self.output_projection(x)
        
        if output_attentions:
            return x, attention
        return x


class feed_forward(nn.Module):
    def __init__(self, conf) -> None:
        super(feed_forward, self).__init__()
        
        self.hidden_dim = conf["hidden_dim"]
        self.intermediate_dim = conf["intermediate_dim"]

        self.input_projection = nn.Linear(in_features=self.hidden_dim, out_features=self.intermediate_dim, bias=True)
        self.output_projection = nn.Linear(in_features=self.intermediate_dim, out_features=self.hidden_dim, bias=True)
        # self.dropout = nn.Dropout(p=conf["dropout"])
        # I don't know if it is necessary this drop out
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x:torch.tensor) -> torch.tensor:
        """Forward method of the feed forward class, the hidden dim increases and the decreases to the normal hidden_dim

        Args:
            x (torch.tensor): input tensor from the mhsa, [batch_size, num_heads, seq_length, head_dim]

        Returns:
            torch.tensor: [batch_size, num_heads, seq_length, head_dim]
        """
        x = self.activation(self.input_projection(x))
        # print(x.shape)
        # the include dropout before the last output_projection
        # x = self.dropout(x)
        x = self.output_projection(x)
        
        return x

class layer_norm(nn.Module):
    def __init__(self, conf) -> None:
        super(layer_norm, self).__init__()

        self.eps = conf["eps"]
        self.hidden_dim = conf["hidden_dim"]

        self.gamma = nn.Parameter(torch.ones(self.hidden_dim))
        self.beta =  nn.Parameter(torch.zeros(self.hidden_dim))

    def forward(self, x:torch.tensor) -> torch.tensor:
        mean = torch.mean(x, dim=-1, keepdim=True)
        variance = torch.sqrt(torch.mean((x-mean)**2, dim=-1, keepdim=True))

        output = (x-mean) / (variance + self.eps)

        output = output * self.gamma + self.beta
        return output
    
class encoder_block(nn.Module):
    def __init__(self, conf) -> None:
        super(encoder_block, self).__init__()

        self.msha = multi_head_self_attention(conf)
        self.ff = feed_forward(conf)
        self.layernorm1 = layer_norm(conf)
        self.layernorm2 = layer_norm(conf)
        # "We apply dropout to the output of each sub-layer, before it is added to the sub-layer input and normalized"
        # self.dropout = conf["dropout"]
        self.dropout = nn.Dropout(p=conf["dropout"])
        

    def forward(self, x:torch.tensor, attention_mask=None) -> torch.tensor:
        hidden_states = self.dropout(self.msha(x, attention_mask))
        # the layer norm is after the residual connection
        x = self.layernorm1(x + hidden_states)

        hidden_states = self.dropout(self.ff(x))
        x = self.layernorm2(x + hidden_states)
        return x

class decoder_block(nn.Module):
    def __init__(self, conf) -> None:
        super(decoder_block, self).__init__()

        self.msha = multi_head_self_attention(conf)
        self.ff = feed_forward(conf)
        self.layernorm1 = layer_norm(conf)
        self.layernorm2 = layer_norm(conf)
        self.layernorm3 = layer_norm(conf)
        
        # self.dropout = conf["dropout"]
        self.dropout = nn.Dropout(p=conf["dropout"])
        
    def forward(self, x:torch.tensor, attention_mask=None, context=None, context_mask=None, output_attentions:bool=False) -> torch.tensor:
        # masked self attention, so here we need to use the causal_attention_mask
        if output_attentions:
            hidden_states, attentions = self.msha(x, attention_mask, None, output_attentions=output_attentions)
            # hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = self.dropout(hidden_states)
        else:
            hidden_states = self.dropout(self.msha(x, attention_mask, None, output_attentions=output_attentions))
        x_1 = self.layernorm1(x + hidden_states)

        # cross attention, here it is not masked so we can use the normal attention mask
        if context != None:
            hidden_states = self.dropout(self.msha(x_1, context_mask, context))
            x = self.layernorm2(x + hidden_states)
        
        # there is a difference between them so the msha is working
        # print(f"Diff between first msha and cross attention in decoder layer: {abs(x_1-x)}")
        
        hidden_states = self.dropout(self.ff(x))
        x = self.layernorm3(x + hidden_states)
        
        # the difference exists
        # print(f"Diff between first msha and ff in decoder layer: {abs(x_1-x)}")
        if output_attentions:
            return x, attentions
        return x
    
class encoder(nn.Module):
    def __init__(self, conf) -> None:
        super(encoder, self).__init__()
        
        self.num_layers = conf["num_layers"]
        self.layers = nn.ModuleList()
        
        for i in range(self.num_layers):
            self.layers.append(encoder_block(conf))
        
    def forward(self, hidden_states:torch.tensor, attention_mask=None) -> torch.tensor:
        
        for i in range(len(self.layers)):
            hidden_states = self.layers[i](hidden_states, attention_mask)

        return hidden_states

class decoder(nn.Module):
    def __init__(self, conf) -> None:
        super(decoder, self).__init__()
        
        self.num_layers = conf["num_layers"]
        self.layers = nn.ModuleList()
        # self.embeddings = embeddings(conf)
        # self.pos_embeddings = positional_encoding(conf)

        
        for i in range(self.num_layers):
            self.layers.append(decoder_block(conf))
            
    def set_embeddings_weigths(self, embeddings):
        self.embeddings.weights.weight = embeddings.weights.weight
        
    def forward(self, hidden_states:torch.tensor, attention_mask=None, context=None, context_mask=None, output_attentions:bool=False) -> torch.tensor:
        # the input is the encoder output, as we will be creating the output in a loop, first token is [EOS]
        # batch_size, seq_length, _ = hidden_states.shape

        # get the embeddings of the token 0, which can be EOS
        # hidden_states = self.embeddings(torch.tensor([0]))
        # get the embeddings of the target_ids
        # hidden_states = self.embeddings(input_ids)
        # hidden_states = self.pos_embeddings(hidden_states)
        
        if output_attentions:
            attentions_array = []
        
        for i in range(len(self.layers)):
            hidden_states, attentions = self.layers[i](hidden_states, attention_mask, context, context_mask, output_attentions=output_attentions)
            attentions_array.append(attentions)

        if output_attentions:
            return hidden_states, attentions_array
        return hidden_states

class transformer(nn.Module):
    def __init__(self, conf) -> None:
        super(transformer, self).__init__()

        self.encoder = encoder(conf)
        self.decoder = decoder(conf)

        self.embeddings = embeddings(conf)
        self.pos_embeddings = positional_encoding(conf)
        self.hidden_dim = conf["hidden_dim"]
        self.vocabulary_size = conf["vocabulary_size"]
        
        # During training, we employed label smoothing of value 𝜖𝑙⁢𝑠=0.1
        self.label_smoothing = conf["label_smoothing"]

        # the output embeddings layer, is the LM Head, a linear with input hidden_dim and output vocab_size
        self.lm_head = nn.Linear(in_features=self.hidden_dim, out_features=self.vocabulary_size, bias=False)
        self.set_tied_embeddings()
        
    def make_causal_mask(self, input_ids_shape, dtype, target_padding_mask, context_length=0):
        batch_size, seq_length = input_ids_shape
        
        # lower diagonal matrix as we want to use the first token in the first sentence
        # there are several ways of computing the mask
        # attention_mask = torch.tril(torch.ones((seq_length, context_length)))
        
        # attention_mask = (1 - attention_mask) * torch.finfo(dtype).min

        # the first mask is just the causal one, without taking into account the context or pass values
        mask = torch.full((seq_length, seq_length), torch.finfo(dtype).min)
        
        mask_cond = torch.arange(mask.size(-1))
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        # print(mask)
        # tensor([[ 1], [ 2], [ 3]])
        
        if context_length > 0:
            mask = torch.cat([torch.zeros(seq_length, context_length), mask], dim=-1)
        
        print(mask.shape)
        mask = mask[None, None, :, :].expand(batch_size, 1, seq_length, seq_length+context_length)
        # the target_padding_mask has ones for the places to attend and 0 for the padding
        # mask = mask.logical_or(target_padding_mask)

        return mask

    def create_causal_mask_with_padding_mult(self, seq_len, padding_mask, device):
        """
        Creates a causal mask with padding taken into account.

        Args:
            seq_len (int): The length of the sequence.
            padding_mask (torch.Tensor): A tensor of shape (batch_size, seq_len) with 0 where padding is present and 1 otherwise.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, 1, seq_len, seq_len) representing the causal mask with padding.
        """
        # Create a causal mask
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.float32, device=device))

        # Combine with padding mask
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
        combined_mask = causal_mask.unsqueeze(0) * padding_mask

        combined_mask = (1-combined_mask) * torch.finfo(torch.float32).min

        return combined_mask
    
    def set_tied_embeddings(self):
        # self.decoder.set_embeddings_weigths(self.embeddings)
        self.lm_head.weight = self.embeddings.weights.weight
    
    def forward(self, src_input_ids:torch.tensor,  tgt_input_ids:torch.tensor, src_attention_mask=None, tgt_attention_mask=None, last_hidden_states:bool=False, output_attentions:bool=False):
        # we need 3 masks and their padding masks, encoder_mask: [seq_len, seq_len], decoder_mask: [target_len, target_len]
        # context_mask: [seq_len, target_len] this is not masked so it can see everything
        hidden_states = self.embeddings(src_input_ids)
        hidden_states = self.pos_embeddings(hidden_states)
        
        # attention_mask for the encoder, after substracting, the 0 should be attented so we mult the 1 to the min value and then add 1 to have the 0 as good values
        dtype = hidden_states.dtype
        if len(src_attention_mask.shape) == 2:
            attention_mask = src_attention_mask.unsqueeze(1).unsqueeze(1)
        attention_mask = (1-attention_mask) * torch.finfo(dtype).min
        # print(f"Attention mask for the encoder: {attention_mask}")
        # encoder
        hidden_states = self.encoder(hidden_states, attention_mask)
        
        # batch_size, context_length = input_ids.shape
        # causal_attention_mask = self.make_causal_mask(input_ids_shape=(batch_size, max_tokens), dtype=dtype, context_length=context_length)
        causal_attention_mask = self.create_causal_mask_with_padding_mult(tgt_attention_mask.shape[1], tgt_attention_mask, device=tgt_input_ids.device)
        # print(f"Attention mask for the decoder: {causal_attention_mask[0][0][-3:]}")

        # decoder
        # cross attention mask, is the encoder mask, as we have QxK: [batch_size, query_seq_len, key_seq_len] and the attention mask is [1, 1, key_seq_len]
        # cross_attention_mask = torch.cat(src_attention_mask, tgt_attention_mask)
        tgt_hidden_states = self.embeddings(tgt_input_ids)
        tgt_hidden_states = self.pos_embeddings(tgt_hidden_states)
        
        if output_attentions:
            hidden_states, attentions_array = self.decoder(tgt_hidden_states, causal_attention_mask, hidden_states, attention_mask, output_attentions=output_attentions)
        else:
            hidden_states = self.decoder(tgt_hidden_states, causal_attention_mask, hidden_states, attention_mask, output_attentions=output_attentions)
        
        logits = self.lm_head(hidden_states)
        
        # now we compute the loss
        criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        # the labels are the inputs shifted to the right, we compute all the tokens, but for the loss we shift them
        # as we are using the causal mask, we don't need to shift the input_ids before computing  
        labels = tgt_input_ids[:, 1:].contiguous()
        shifted_logits = logits[..., :, :-1, :].contiguous()
        # print(labels)
        # print(shifted_logits.shape)
        
        # we need to have only 2 dimensions for the crossentropy, so shifted_logits: [batch_size * seq_length, vocab_size] and labels: [batch_size * seq_length]
        shifted_logits = shifted_logits.view(-1, self.vocabulary_size)
        labels = labels.view(-1)
        loss = criterion(shifted_logits, labels)
        
        if last_hidden_states:
            if output_attentions:
                return logits, loss, hidden_states, attentions_array
            else:
                return logits, loss, hidden_states
            
        if output_attentions:
            return logits, loss, attentions_array

        return logits, loss
    
    def generate(self, src_input_ids:torch.Tensor, tgt_max_length, src_attention_mask=None, do_sample:bool=True, top_k:int=20, first_token:int=None, debug:bool=True):
        hidden_states = self.embeddings(src_input_ids)
        hidden_states = self.pos_embeddings(hidden_states)
        
        # attention_mask for the encoder, after substracting, the 0 should be attented so we mult the 1 to the min value and then add 1 to have the 0 as good values
        dtype = hidden_states.dtype
        if len(src_attention_mask.shape) == 2:
            attention_mask = src_attention_mask.unsqueeze(1).unsqueeze(1)
        elif src_attention_mask == None:
            attention_mask = torch.ones_like(src_input_ids).unsqueeze(1).unsqueeze(1)
        attention_mask = (1-attention_mask) * torch.finfo(dtype).min
        
        hidden_states = self.encoder(hidden_states, attention_mask)
        
        # batch_size, context_length = input_ids.shape
        # causal_attention_mask = self.make_causal_mask(input_ids_shape=(batch_size, max_tokens), dtype=dtype, context_length=context_length)
        # tgt_input_ids = torch.ones((src_input_ids.shape[0], tgt_max_length)).to(src_input_ids.device)
        # [PAD] id is 3 so the first token is [PAD]
        if first_token != None:
            tgt_input_ids = torch.tensor([first_token], dtype=src_input_ids.dtype, device=src_input_ids.device).unsqueeze(0)
        else:
            tgt_input_ids = torch.tensor([1000], dtype=src_input_ids.dtype, device=src_input_ids.device).unsqueeze(0)
        # default_attention_mask = torch.ones(tgt_input_ids.shape[:2], dtype=torch.long, device=src_input_ids.device)
        
        if debug:
            outputs_array = []
        # print(tgt_input_ids.device)
        for i in range(1, tgt_max_length-1):
            tgt_hidden_states, causal_attention_mask = self.prepare_inputs_for_generation(tgt_input_ids)
            
            tgt_hidden_states = self.decoder(tgt_hidden_states, causal_attention_mask, hidden_states, attention_mask)

            logits = self.lm_head(tgt_hidden_states)
            # logits [batch_size, tgt_seq_length, vocab_size], we only want the 
            next_token_logits = logits[:, -1, :]
            if do_sample:
                topk_scores = torch.topk(next_token_logits, dim=-1, k=top_k)
                indices_to_remove = next_token_logits < topk_scores[0][..., -1, None]
                scores_processed = next_token_logits.masked_fill(indices_to_remove, torch.finfo(next_token_logits.dtype).min)
                scores = F.softmax(scores_processed, dim=-1)
                if debug:
                    outputs_array.append(tgt_hidden_states)
                    print(torch.topk(scores, dim=-1, k=top_k))
                next_token = torch.multinomial(scores, num_samples=1)
            else:
                next_token = next_token_logits.argmax(dim=-1)
            
            tgt_input_ids = torch.concat([tgt_input_ids, next_token], dim=-1)
            # if next_token.item() == 3:
            #     break
        if debug:
            return tgt_input_ids, tgt_hidden_states
        
        return tgt_input_ids
                
    def prepare_inputs_for_generation(self, inputs_ids:torch.tensor):
        tgt_hidden_states = self.embeddings(inputs_ids)
        tgt_hidden_states = self.pos_embeddings(tgt_hidden_states)
        
        tgt_attention_mask = torch.ones((inputs_ids.shape[0], inputs_ids.shape[1]), device=inputs_ids.device)
        causal_attention_mask = self.create_causal_mask_with_padding_mult(tgt_attention_mask.shape[1], tgt_attention_mask, device=inputs_ids.device)
        return tgt_hidden_states, causal_attention_mask
    
if __name__ == "__main__":
    seq_length = 32
    x = torch.randn((1, seq_length, 512))
    attention_mask = torch.ones((1, 1, seq_length, seq_length))
    context = torch.randn((1, int(seq_length/2), 512))
    input_ids = torch.arange(8).unsqueeze(0)
    attention_mask = torch.ones(input_ids.shape)
    target_ids = torch.arange(16).unsqueeze(0)
    tgt_attention_mask = torch.ones(target_ids.shape)

    conf = {"hidden_dim": 512, "vocabulary_size": 12000, "num_heads": 8, "intermediate_dim": 2048, "eps": 1e-06, "num_layers": 6, "dropout": 0.1,
            "label_smoothing": 0.1, "max_length": 128}
    pe = positional_encoding(conf)

    encodings = pe.generate_encoding(seq_length)
    print(pe.encoding.shape)
    plot = plt.matshow(pe.encoding)
    plt.gcf().colorbar(plot)
    plt.show()

    import numpy as np
    conf["hidden_dim"]= 20
    pe_4 = positional_encoding(conf)
    plt.plot(np.arange(100), pe_4.encoding[:100, 4:8].data.numpy())
    plt.legend(["dim %d"%p for p in [4,5,6,7]])
    plt.show()
    # embedds = embeddings(conf)
    # print(embedds.weights.shape)
    # print(embedds(torch.arange(12).unsqueeze(0)).shape)
    
    # msa = multi_head_self_attention(conf)

    # attention = msa(x)
    # print(attention.shape)

    # ff = feed_forward(conf)
    # output = ff(x)
    # print(output.shape)

    # ln = layer_norm(conf)
    # output = ln(x)
    # print(output.shape)
    # print(x[0, 0, 0:10])
    # print(output[0, 0, 0:10])

    # mean = torch.mean(x[0, 0], dim=-1, keepdim=True)
    # variance = torch.sqrt(torch.mean((x[0, 0]-mean)**2, dim=-1, keepdim=True))
    # print(mean, variance)
    # mean = torch.mean(output[0, 0], dim=-1, keepdim=True)
    # variance = torch.sqrt(torch.mean((output[0, 0]-mean)**2, dim=-1, keepdim=True))
    # print(mean, variance)

    # en_b = encoder_block(conf)
    # print(en_b(x, attention_mask).shape)
    # dec_b = decoder_block(conf)
    # print(dec_b(x, attention_mask, context).shape)

    # enc = encoder(conf)
    # print(enc(x, attention_mask).shape)

    # dec = decoder(conf)
    # print(dec(x, attention_mask, context).shape)
    
    # import torchinfo
    
    # model = transformer(conf)
    # outputs, loss = model(input_ids, target_ids, attention_mask, tgt_attention_mask)
    # causal_mask = model.make_causal_mask((1, 16), torch.float16, 8)
    # print(causal_mask.shape)
    # print(causal_mask[0][0][-3:])
    # print(outputs.shape)
    # print(loss)
    # print(torchinfo.summary(model, depth=15))