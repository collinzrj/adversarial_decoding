import torch
from transformers import BertModel, BertForSequenceClassification,  BertForNextSentencePrediction, BertForMaskedLM
from transformers.models.bert.modeling_bert import BertEmbeddings

from bert_layers import BertEncoderPast


class BertForConcatNextSentencePrediction(BertForNextSentencePrediction):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertConcatModel(config)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            next_sentence_label=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        seq_relationship_score = self.cls(pooled_output)
        # add hidden states and attention if they are here
        outputs = (seq_relationship_score, pooled_output) + outputs[2:]
        if next_sentence_label is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            outputs = (next_sentence_loss,) + outputs

        # (next_sentence_loss), seq_relationship_score,
        # (hidden_states), (attentions)
        return outputs


class BertForConcatSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertConcatModel(config)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits, pooled_output) + outputs[2:]

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = torch.nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs


class BertConcatEmbeddings(BertEmbeddings):
    def forward(self, input_ids=None, token_type_ids=None, position_ids=None,
                inputs_embeds=None):
        if input_ids is not None and inputs_embeds is not None:
            input_shape = (input_ids.size(0), input_ids.size(1) + inputs_embeds.size(1))
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long,  device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        elif input_ids is not None:
            inputs_a_embeds = self.word_embeddings(input_ids)
            inputs_embeds = torch.cat([inputs_a_embeds, inputs_embeds], dim=1)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertConcatModel(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertConcatEmbeddings(config)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
    ):
        past_length = 0
        if input_ids is not None and inputs_embeds is not None:
            input_shape = (input_ids.size(0), input_ids.size(1) + inputs_embeds.size(1))
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        batch_size, seq_length = input_shape
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length + past_length), device=device)  # (bs, seq_length)

        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(past_length + seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(
                    batch_size, past_length + seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(attention_mask.dtype)
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape))

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to
        # [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            elif encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
            else:
                raise ValueError("Wrong shape for encoder_hidden_shape (shape {}) or "
                                 "encoder_attention_mask (shape {})".format(encoder_hidden_shape,
                                                                            encoder_attention_mask.shape))

            # fp16 compatibility
            encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=next(self.parameters()).dtype)
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1))
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        position_ids = torch.arange(past_length, past_length + seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)

        if input_ids is not None and inputs_embeds is not None:
            if token_type_ids is None:
                input_a_shape = input_ids.size()
                token_a_type_ids = torch.zeros(input_a_shape, dtype=torch.long, device=device)
                input_b_shape = inputs_embeds.size()[:-1]
                token_b_type_ids = torch.ones(input_b_shape, dtype=torch.long, device=device)
                token_type_ids = torch.cat([token_a_type_ids, token_b_type_ids], dim=1)

            embedding_output = self.embeddings(
                input_ids=input_ids, position_ids=position_ids,
                token_type_ids=token_type_ids, inputs_embeds=inputs_embeds,
            )
        else:
            embedding_output = self.embeddings(
                input_ids=input_ids, position_ids=position_ids,
                token_type_ids=token_type_ids, inputs_embeds=inputs_embeds,
            )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        return outputs


class BertAutoRegressiveModel(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = BertEncoderPast(config)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past=None
    ):
        if past is None:
            past_length = 0
        else:
            past_length = past[0][0].size(-2)

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        batch_size, seq_length = input_shape
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length + past_length), device=device)  # (bs, seq_length)
        seq_ids = torch.arange(past_length + seq_length, device=device)
        # add a upper triangle mask for auto-regressive language model
        causal_mask = seq_ids[None, None, :].repeat(batch_size, past_length + seq_length, 1)  <= seq_ids[None, :, None]
        causal_mask = causal_mask.to(attention_mask.dtype)
        extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape
        # [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1))
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        position_ids = torch.arange(past_length, past_length + seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids,
            token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past=past
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        # sequence_output, pooled_output, (hidden_states), (attentions)
        return outputs


class BertForLM(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        print(config)
        config.output_hidden_states = True
        self.bert = BertAutoRegressiveModel(config)
        self.start_idx = 1
        self.init_weights()

    def prepare_inputs_for_generation(self, input_ids, past):
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
        return {"input_ids": input_ids, "past": past}

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            masked_lm_labels=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            one_hot_labels=None,
            past=None
    ):
        label_start_idx = 1
        if inputs_embeds is not None:
            start_embeds = self.get_input_embeddings().weight[self.start_idx]
            inputs_embeds = torch.cat([start_embeds.view(1, 1, -1), inputs_embeds], 1)
            label_start_idx = 0

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past=past
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        # Add hidden states and attention if they are here
        outputs = (prediction_scores,) + outputs[2:]

        # we are doing next-token prediction;
        # shift prediction scores and input ids by one
        if one_hot_labels is not None:
            prediction_scores = prediction_scores[:, :-1, :].contiguous()
            lm_labels = one_hot_labels[:, label_start_idx:, :].contiguous()
            nll = -torch.log_softmax(prediction_scores, -1)
            ltr_lm_loss = torch.sum(nll * lm_labels, -1).mean()
            outputs = (ltr_lm_loss,) + outputs
        elif labels is not None:
            prediction_scores = prediction_scores[:, :-1, :].contiguous()
            lm_labels = labels[:, label_start_idx:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            ltr_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), lm_labels.view(-1))
            outputs = (ltr_lm_loss,) + outputs
        return outputs
