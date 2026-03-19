from header import *

from .openllama import (
    OpenLLAMAPEFTModel,
    encode_text_reference,
    encode_text_with_prompt_ensemble,
    process_batch_instance,
)


class OpenLLAMAPEFTModelMultiClsAuxNoGT(OpenLLAMAPEFTModel):
    """
    Multiclass + auxiliary-branch variant without GT supervision from:
    - fake_image_box (bbox/giou losses removed)
    - fake_text_pos (token-position supervision not used)
    """

    def forward(self, inputs):
        # Obtain visual features.
        image_paths = inputs["images"]
        img_embeds, _, patch_tokens, img_embeds_before_proj = self.encode_image_from_tensor(image_paths)
        img_patch_feaure_layers = torch.stack(patch_tokens, dim=0)
        img_patch_feaure = torch.mean(img_patch_feaure_layers, dim=0)
        img_all_feature = torch.cat([img_embeds_before_proj, img_patch_feaure], dim=1)

        # Use caption text only; do not use fake_text_pos supervision signals.
        text = inputs["captions"]
        bs = img_embeds.shape[0]

        feats_text_tensor = encode_text_with_prompt_ensemble(
            self.visual_encoder, ["object" for _ in range(bs)], self.device
        )
        news_text_embeds, news_text_patch_embeds, _ = encode_text_reference(
            self.visual_encoder, text, self.device
        )
        text_all_feature = torch.cat([news_text_embeds, news_text_patch_embeds], dim=1)

        # Keep auxiliary branches to produce forgery prompts, but no GT losses.
        forgery_embed, forgery_patch_embeds = self.Cross_Modal_Reason(bs, img_all_feature, text_all_feature)
        _, atts_local_feat_aggr = self.Bbox_Verification(forgery_patch_embeds)
        forgery_map_prompts, _ = self.Segmentation_Verification(
            forgery_patch_embeds,
            feats_text_tensor,
            atts_cls_feat=forgery_embed,
            atts_bbox_feat=atts_local_feat_aggr,
        )

        output_texts = inputs["texts"]
        input_ids, target_ids, attention_mask = process_batch_instance(
            self.llama_tokenizer, output_texts, self.max_tgt_len
        )
        inputs_embeds, targets, attention_mask = self.prompt_wrap(
            img_embeds, input_ids, target_ids, attention_mask, forgery_map_prompts
        )
        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss

        chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]
        labels = targets[:, 2:]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask
        gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()

        return loss, gen_acc
