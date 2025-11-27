from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from datasets import load_dataset

def to_triplet(example):
    return {
        "prompt": (example["conversations"][0].get("value")),
        "chosen": (example["chosen"].get("value") or "").strip(),
        "rejected": (example["rejected"].get("value") or "").strip(),
    }

# DPODataset用于读取单条样本转换为token id序列
class DPODataset(Dataset):
    def __init__(self, data, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.datas = data

    def __getitem__(self, index):
        sample = self.datas[index]
        prompt = sample['prompt']
        chosen = sample['chosen']
        rejected = sample['rejected']
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompt_inputs = self.tokenizer(text=text)['input_ids']
        rejected_inputs = self.tokenizer(text=rejected)['input_ids'] + [self.tokenizer.eos_token_id]
        chosen_inputs = self.tokenizer(text=chosen)['input_ids'] + [self.tokenizer.eos_token_id]
        return [prompt_inputs, chosen_inputs, rejected_inputs]

    def __len__(self):
        return len(self.datas)

# DPODataCollator用于将数据打包为batch的形式
class DPODataCollator:
    def __init__(self, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, features):
        inputs_ids = []
        labels = []

        # features: {prompt_input_ids, chosen_input_ids, rejected_input_ids}
        for feature in features:
            inputs_ids.append(feature[0] + feature[1])
            labels.append([0]*len(feature[0]) + feature[1])

        for feature in features:
            inputs_ids.append(feature[0] + feature[2])
            labels.append([0]*len(feature[0]) + feature[2])

        def process(inputs_ids, labels):
            inputs_ids = [input_ids[:self.max_seq_len] for input_ids in inputs_ids]
            labels = [label[:self.max_seq_len] for label in labels]
            max_len = max([len(input_ids) for input_ids in inputs_ids])
            batch_input_ids = []
            batch_labels = []

            for input_ids, label in zip(inputs_ids, labels):
                if len(input_ids) <= max_len:
                    input_ids = input_ids+[0]*(max_len-len(input_ids))
                    label = label+[0]*(max_len-len(label))
                    batch_input_ids.append(input_ids[:-1])
                    batch_labels.append(label[1:])
            return batch_input_ids, batch_labels

        inputs_ids, labels = process(inputs_ids, labels)

        return {
                "input_ids": torch.tensor(inputs_ids),
                "labels": torch.tensor(labels)
            }

def logits_to_probs(logits, labels):
    # logits shape: (batch_size, seq_len, vocab_size)
    # labels shape: (batch_size, seq_len)
    # probs shape: (batch_size, seq_len)
    log_probs = F.log_softmax(logits, dim=2)
    probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1) # 每个位置把对应token的概率从vocab_size中挑出来
    return probs

def mask_logits(log_probs, labels):
    # log_probs shape: (2*batch_size, seq_len)
    # labels_masks shape: (2*batch_size)
    seq_probs = []
    for log_prob, label in zip(log_probs, labels):
        seq_probs.append(log_prob[label != 0].sum().unsqueeze(0))

    return seq_probs

def dpo_loss(ref_probs, probs, beta):
    def split_probs(probs):
        # label中包含chosen和rejected, 因此生成部分的概率手动拆成chosen序列和rejected序列
        len_chosen = int(len(probs) // 2)
        chosen_data = probs[:len_chosen]
        reject_data = probs[len_chosen:]
        return torch.cat(chosen_data), torch.cat(reject_data)

    # 分别计算ref_model和policy_model, chosen和rejected序列的对数似然概率
    ref_chosen_probs, ref_reject_probs = split_probs(ref_probs)
    chosen_probs, reject_probs = split_probs(probs)

    # Bradley-Terry模型的核心公式, 因为都是对数似然概率, 除法转换成减法
    # (chosen_probs - ref_chosen_probs) - (rejected_probs - ref_reject_probs)
    pi_logratios = chosen_probs - reject_probs
    ref_logratios = ref_chosen_probs - ref_reject_probs
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta*logits)
    return loss.mean()

class DPOTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs['input_ids']
        labels = inputs['labels']

        # 计算ref_modell生成部分的概率
        with torch.no_grad():
            ref_logits = ref_model(input_ids=input_ids, labels=labels).logits
        # 2*batch_size * seq_len * vacab_size -> 2*batch_size * seq_len
        # 这一步是把原始的logits分数转换为取labels中各token的对数似然概率
        ref_log_probs = logits_to_probs(ref_logits, labels)
        # 2*batch_size * seq_len -> 2*batch_size
        # 这一步是把prompt遮住, 并且计算生成序列的对数似然概率
        ref_seq_probs = mask_logits(ref_log_probs, labels)

        # 计算polilcy_model生成部分的概率, 同ref_model对数似然概率计算的过程
        policy_logits = model(input_ids=input_ids, labels = labels).logits
        policy_log_probs = logits_to_probs(policy_logits, labels)
        policy_seq_probs = mask_logits(policy_log_probs, labels)

        # 计算loss
        loss = dpo_loss(ref_seq_probs, policy_seq_probs, 0.1)
        return loss

if __name__ == "__main__":
    model_path = "./Qwen2.5-0.5B"
    dataset_path = "./COIG-P/data/*.parquet"
    output_path = "./Qwen2.5-0.5B-DPO"

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    policy_model = AutoModelForCausalLM.from_pretrained(model_path)
    ref_model = AutoModelForCausalLM.from_pretrained(model_path).eval().to('cuda')

    raw = load_dataset("parquet", data_files=dataset_path)["train"]
    train_split = raw.train_test_split(test_size=0.9, seed=42)["train"]
    triplet = train_split.map(to_triplet, remove_columns=raw.column_names)
    dataset = DPODataset(triplet, tokenizer)

    data_collator = DPODataCollator(tokenizer, max_seq_len=256)
    args = TrainingArguments(output_dir=output_path,
                            num_train_epochs=1,
                            do_train=True,
                            per_device_train_batch_size=8,
                            gradient_accumulation_steps=2,
                            logging_steps=50,
                            report_to='tensorboard',
                            save_total_limit=3,
                            bf16=True,
                            learning_rate=1e-5,
                            lr_scheduler_type='cosine',
                            dataloader_num_workers=1,
                            dataloader_pin_memory=True,
                            save_safetensors=False,
                            save_steps=100)

    trainer = DPOTrainer(
        model=policy_model,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # 如果是初次训练resume_from_checkpoint为false，接着checkpoint继续训练，为True
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model()
    trainer.save_state()