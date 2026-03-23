import pickle
import torch
import sys
import os
from pathlib import Path

# Important: We need raven_cnn's classes to be available for pickle to load
# But instead of relying on pickle.load searching for __main__.TextCNN
# which is brittle, it's safer to just provide the exact same class structure
# or if raven_cnn was exported successfully, let's load it.

sys.path.insert(0, str(Path(__file__).parent))

class CNNClassifier:
    """Wrapper for the exported CNN Model."""
    def __init__(self, model_dict):
        self.model = model_dict["model"]
        self.tok = model_dict["tok"]
        self.intent_encoder = model_dict["intent_encoder"]
        self.tag_vocab = model_dict["tag_vocab"]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
    @classmethod
    def load(cls, path):
        # We must import from raven_cnn so pickle can find classes
        import raven_cnn
        import sys
        # Map classes into __main__ as pickle treats it as the default namespace
        # when models are saved from direct execution scripts (like in Colab)
        sys.modules["__main__"].TextCNN = raven_cnn.TextCNN
        sys.modules["__main__"].Tokenizer = raven_cnn.Tokenizer
        
        with open(path, "rb") as f:
            model_dict = pickle.load(f)
        return cls(model_dict)
        
    def predict(self, text):
        import raven_cnn
        
        # ── Encode text ──
        ids = torch.tensor([self.tok.encode(text)], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            intent_logits, tag_logits = self.model(ids)
            
        # ── Intent ──
        intent_proba = torch.softmax(intent_logits[0], dim=0).cpu().numpy()
        best_idx = int(intent_proba.argmax())
        intent = self.intent_encoder.classes_[best_idx]
        conf = float(intent_proba[best_idx])
        
        # ── Tags: model + rules ──
        tag_probs = torch.sigmoid(tag_logits).squeeze(0).cpu().numpy()
        model_tags = [self.tag_vocab[i] for i, p in enumerate(tag_probs) if p > 0.35]
        
        keyword_tag_str = raven_cnn.enrich_tags_from_text(text, "")
        keyword_tags = [t.strip() for t in keyword_tag_str.split(",") if t.strip() not in ("", "untagged")]
        merged = list(dict.fromkeys(model_tags + keyword_tags))
        pred_tags = raven_cnn.resolve_conflicts(merged)
        
        # ── Lang ──
        lang = raven_cnn.detect_language(text)
        
        return {
            "intent": intent,
            "confidence": conf,
            "tags": pred_tags,
            "lang": lang
        }
