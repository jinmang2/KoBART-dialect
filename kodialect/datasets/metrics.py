import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_cls_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions[0].argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'macro_f1': f1,
        'macro_precision': precision,
        'macro_recall': recall
    }


def make_compute_metrics(tokenizer, classifier, args):
    classifier = classifier.to(args.device)

    def compute_metrics(pred, tokenizer=tokenizer, classifier=classifier):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        tgt = []
        for pred in preds:
            e = torch.arange(len(pred))[pred.eq(tokenizer.eos_token_id)]
            e = e[0] if 0 < len(e) and e[0] < 30 else 30
            tgt.append(pred[:e].cpu().tolist())
        tgt = tokenizer.pad({"input_ids":tgt}, return_tensors="pt")
        tgt_idx = tgt["input_ids"].to(self.args.device)
        y_hat = classifier(tgt).logits.detach().argmax(dim=-1)
        acc = sum(y_hat == labels)
        return {
            "accuracy": acc,
        }

    return compute_metrics
