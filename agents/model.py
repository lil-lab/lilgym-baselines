# Based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail

import numpy as np
import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import ViltProcessor, ViltModel

from .distributions import Categorical, MultiCategorical
from .util import init

from functools import reduce
import operator


text_feats_dim = {
    "bertfix": 768,
}


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(-1, reduce(operator.mul, x.size()[1:], 1))


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None, custom_model=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                if custom_model == "vilt":
                    base = ViLT2Paths
                if custom_model == "c3bert":
                    base = CBert2Paths
            else:
                raise NotImplementedError
        
        if isinstance(obs_shape, tuple):
            self.base = base(obs_shape[0], **base_kwargs)
        elif isinstance(obs_shape, dict):
            self.base = base(obs_shape["image"][0], **base_kwargs)
        
        if action_space.__class__.__name__ in ["Discrete"]:
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ in ['TowerActionSpace', 'ScatterActionSpace']:
            latent_dim = self.base.latent_dim
            
            nvec = action_space.actions_dim
            base_dim = nvec[0]
            self.dist_base = Categorical(latent_dim, base_dim)
            if len(nvec) == 3:
                self.env_opt = "tower"
                pos_dim = nvec[1]
                att_dim = nvec[2]
                self.dist_pos = Categorical(latent_dim, pos_dim)
                self.dist_att = Categorical(latent_dim, att_dim)
            elif len(nvec) == 6:
                self.env_opt = "scatter"
                pos_dim = nvec[1:3]
                att_dim = nvec[3:]
                self.dist_pos = MultiCategorical(latent_dim, pos_dim)
                self.dist_att = MultiCategorical(latent_dim, att_dim)
        else:
            raise NotImplementedError
        
    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False, random=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)

        # Separating action_type (ADD, REMOVE, STOP) and other action_att
        # for multi action invalid masking
        dist_base = self.dist_base(actor_features)
        if random:
            action_base_np = np.expand_dims(np.random.randint(0, 3, size=1), axis=0)
            action_base = torch.tensor(action_base_np)
        else:
            if deterministic:
                action_base = dist_base.mode()
            else:
                action_base = dist_base.sample()

        action_base_log_probs = dist_base.log_probs(action_base)

        action, action_log_probs = self.sample_action_components(action_base, 
                                                                      action_base_log_probs, 
                                                                      actor_features,
                                                                      deterministic=deterministic)

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value
    
    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist_base = self.dist_base(actor_features)

        action_base = torch.unsqueeze(action[:,0], 1)
        action_base_log_probs = dist_base.log_probs(action_base)
        dist_base_entropy = dist_base.entropy().mean()

        action_log_probs, dist_entropy = self.get_action_components_batch(action_base, 
                                                                            action_base_log_probs, 
                                                                            dist_base_entropy,
                                                                            actor_features,
                                                                            action)
        return value, action_log_probs, dist_entropy, rnn_hxs

    def sample_action_components(self, 
                                 action_base, 
                                 action_base_log_probs, 
                                 actor_features, 
                                 deterministic=False):
        # With masking or with no masking but separating action_type and ation_att 
        if action_base == 0: # STOP
            action = action_base
            action_log_probs = action_base_log_probs
        else: # REMOVE, or ADD
            dist_pos = self.dist_pos(actor_features)
            if deterministic:
                action_pos = dist_pos.mode()
            else:
                action_pos = dist_pos.sample()
            action_pos_log_probs = dist_pos.log_probs(action_pos)

            if action_base == 2:
                action = torch.cat((action_base, action_pos), 1)
                action_log_probs = torch.add(action_base_log_probs, action_pos_log_probs)
            elif action_base == 1:
                dist_att = self.dist_att(actor_features)
                if deterministic:
                    action_att = dist_att.mode()
                else:
                    action_att = dist_att.sample()
                action_att_log_probs = dist_att.log_probs(action_att)
                action_log_probs_tmp = torch.add(action_base_log_probs, action_pos_log_probs)
                action_log_probs = torch.add(action_log_probs_tmp, action_att_log_probs)

                action = torch.cat((action_base, action_pos, action_att), 1)
        return action, action_log_probs
    
    def get_action_components_batch(self,
                                    action_base, 
                                    action_base_log_probs,
                                    dist_base_entropy,
                                    actor_features,
                                    action):
        ent = dist_base_entropy.item()

        if len(action[0]) > 3: # scatter
            raw_action_pos = action[:,1:3]
            raw_action_att = action[:,3:]
        else: # tower
            raw_action_pos = torch.unsqueeze(action[:,1], 1)
            raw_action_att = torch.unsqueeze(action[:,2], 1)

        valid_indexes_pos = (raw_action_pos[:,0] != -1).nonzero(as_tuple=True)
        valid_indexes_att = (raw_action_att[:,0] != -1).nonzero(as_tuple=True)

        # get the valid probs
        if len(valid_indexes_pos[0]) > 0:
            valid_action_pos = torch.index_select(raw_action_pos, 0, valid_indexes_pos[0])
            dist_pos = self.dist_pos(actor_features[valid_indexes_pos])
            valid_action_pos_log_probs = dist_pos.log_probs(valid_action_pos)

            dist_pos_entropy = dist_pos.entropy().mean()
            ent += dist_pos_entropy.item()
        
        if len(valid_indexes_att[0]) > 0:
            valid_action_att = torch.index_select(raw_action_att, 0, valid_indexes_att[0])
            dist_att = self.dist_att(actor_features[valid_indexes_att])    
            valid_action_att_log_probs = dist_att.log_probs(valid_action_att)

            dist_att_entropy = dist_att.entropy().mean()
            ent += dist_att_entropy.item()

        # put the probs back to a (64, 1) tensor
        action_pos_log_probs = torch.zeros(action_base.shape).to(raw_action_pos.device)
        action_att_log_probs = torch.zeros(action_base.shape).to(raw_action_pos.device)

        if len(valid_indexes_pos[0]) > 0:
            action_pos_log_probs[valid_indexes_pos] = valid_action_pos_log_probs
        if len(valid_indexes_att[0]) > 0:
            action_att_log_probs[valid_indexes_att] = valid_action_att_log_probs

        # sum
        action_log_probs_tmp = torch.add(action_base_log_probs, action_pos_log_probs)
        action_log_probs = torch.add(action_log_probs_tmp, action_att_log_probs)

        # entropy
        dist_entropy = torch.tensor(ent)
        return action_log_probs, dist_entropy


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size


class ViLT2Paths(NNBase):
    def __init__(self, 
                 num_img_inputs=None, 
                 recurrent=False, 
                 hidden_size=None, 
                 text_feat="",
                 learn_opt='',
                 env_opt='',
                 eval_mode=False):
        super(ViLT2Paths, self).__init__(recurrent, hidden_size, hidden_size)

        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
        self.model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
        self.model_critic = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
        self.viltconfig = self.model.config

        self.latent_dim = 768

        self.critic = nn.Sequential(
            nn.Linear(self.viltconfig.hidden_size, self.viltconfig.hidden_size * 2),
            nn.LayerNorm(self.viltconfig.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.viltconfig.hidden_size * 2, 1),
            nn.Tanh(),
        )

        if eval_mode:
            self.model.eval()
        
    def get_target_str(self, tgt):
        if tgt == 1:
            return "True"
        elif tgt == 0:
            return "False"
    
    def forward(self, inputs, rnn_hxs, masks):
        sentences = [s for s in inputs["sentence"]]
        targets = [self.get_target_str(t) for t in inputs["target"]]
        imgs = [i.cpu() for i in inputs["image"]] # cpu for processor processing

        sentences = [s + " <TARGET> " + targets[i] for i, s in enumerate(sentences)]
        
        processed_inputs = self.processor(imgs, sentences, return_tensors="pt", padding=True).to("cuda")

        outputs = self.model(**processed_inputs)
        outputs_critic = self.model_critic(**processed_inputs)
        
        x = outputs.pooler_output
        x_critic = outputs_critic.pooler_output
        
        return self.critic(x_critic), x, rnn_hxs


class CBert2Paths(NNBase):
    def __init__(self, 
                 num_img_inputs, 
                 recurrent=False, 
                 hidden_size=512, 
                 text_feat="bpe",
                 learn_opt='',
                 env_opt='',
                 eval_mode=False):
        super(CBert2Paths, self).__init__(recurrent, hidden_size, hidden_size)

        self.learn_opt = learn_opt

        self.latent_dim = hidden_size
        
        self.text_feat = text_feat
        self.text_feat_dim = text_feats_dim[text_feat]
        self.text_tgt_dim = self.text_feat_dim
        
        self.tgt_dim = 32
        self.text_tgt_dim += self.tgt_dim
        self.tgt_embeds = nn.Embedding(2, 32)
    
        if self.text_feat == "bertfix":
            config = AutoConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.bertmodel = AutoModel.from_pretrained('bert-base-uncased', config=config)
            # Using the same configuratoin as in flairNLP
            self.layer_mean = True
            self.pooling = "mean"
            if self.text_feat == "bertfix":
                self.bertmodel.eval()
                self.sent_to_fixed_embed = {}
            self.sent_to_fixed_tok = {}
    
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))
        
        self.model = nn.Sequential(
            init_(nn.Conv2d(num_img_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 2 * 20, hidden_size)), nn.ReLU())

        self.join_img_sent_actor = nn.Sequential(
            init_(nn.Linear(hidden_size + self.text_tgt_dim, hidden_size)), nn.LeakyReLU(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.LeakyReLU())
        
        self.join_img_sent_critic = nn.Sequential(
            init_(nn.Linear(hidden_size + self.text_tgt_dim, hidden_size)), nn.LeakyReLU(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.LeakyReLU())
        
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = nn.Sequential(
            init_(nn.Linear(hidden_size, 1)), nn.Tanh())

        if eval_mode:
            self.main.eval()
            self.join_img_sent_actor.eval()
            self.join_img_sent_critic.eval()
            self.critic_linear.eval()

    def forward(self, inputs, rnn_hxs, masks):
        img, target = inputs["image"], inputs["target"]

        sent = []        
        if self.text_feat == "bertfix":
            input_sentences = [s for s in inputs["sentence"]]
            sent = []
            
            for sentence in input_sentences:
                if sentence not in self.sent_to_fixed_embed.keys():
                    tok_sentence = self.sent_to_fixed_tok[sentence]
                    with torch.no_grad():
                        outputs = self.bertmodel(**tok_sentence)
                    hidden_states = torch.stack(outputs["hidden_states"])
                    _sent = self.compute_bert_embedding(hidden_states, tok_sentence)
                    sent.append(torch.squeeze(_sent))
                else:
                    sent.append(self.sent_to_fixed_embed[sentence])
            sent = torch.squeeze(torch.stack(sent), 1)
        
        img = self.model(img / 255.0)

        tgt = self.tgt_embeds(torch.tensor(target, dtype=torch.long))
        x = torch.cat((sent, img, tgt), axis=1)

        x_actor = self.join_img_sent_actor(x)
        x_critic = self.join_img_sent_critic(x)

        return self.critic_linear(x_critic), x_actor, rnn_hxs

    def precompute_bert_embedding(self, sentence):
        if sentence not in self.sent_to_fixed_embed.keys():
            tok_sentences = self.tokenizer(sentence, return_tensors="pt", padding=True).to(self.bertmodel.device)
            if self.text_feat == "bertfix":
                with torch.no_grad():
                    outputs = self.bertmodel(**tok_sentences)
            hidden_states = torch.stack(outputs["hidden_states"])
            sent = self.compute_bert_embedding(hidden_states, tok_sentences)
            self.sent_to_fixed_embed[sentence] = sent
    
    def precompute_bert_tok(self, sentence):
         if sentence not in self.sent_to_fixed_tok.keys():
             tok_sentence = self.tokenizer(sentence, return_tensors="pt", padding=True).to(self.bertmodel.device)
             self.sent_to_fixed_tok[sentence] = tok_sentence

    def compute_bert_embedding(self, hidden_states, tok_sentences):
        # Based on flairNLP.
        batch_embedding = []
        layer_indexes = [int(x) for x in range(len(hidden_states))]
        
        for sentence_idx, input_ids in enumerate(tok_sentences['input_ids']):
            sentence_hidden_state = hidden_states[:, sentence_idx, ...]
            
            word_embeddings_in_sentence = []    
            
            start_idx = 1
            end_idx = (input_ids == 102).nonzero(as_tuple=True)[0].item()
            # discard [CLS] and [SEP]
            for subword_start_idx in range(start_idx, end_idx):
                subword_end_idx = subword_start_idx + 1
                subtoken_embeddings = []
                
                for layer in layer_indexes:
                    current_embeddings = sentence_hidden_state[layer][subword_start_idx:subword_end_idx]
                    final_embedding = current_embeddings[0]
                    subtoken_embeddings.append(final_embedding)

                # use layer mean
                if self.layer_mean:
                    sm_embeddings = torch.mean(torch.stack(subtoken_embeddings, dim=1), dim=1)
                    sm_embeddings_list = [sm_embeddings]
                curr_word_embeddings = torch.cat(sm_embeddings_list)
                word_embeddings_in_sentence.append(curr_word_embeddings.unsqueeze(0))
        
            # sentence-level
            word_embeddings = torch.cat(word_embeddings_in_sentence, dim=0)

            if self.pooling == "mean":
                pooled_embedding = torch.mean(word_embeddings, 0)
            batch_embedding.append(pooled_embedding)
        
        batch_embedding = torch.stack(batch_embedding)

        return batch_embedding