import torch

class ClinicalTranslator:
    def __init__(self, backend="google"):
        """
        Initializes the translation backend.
        :param backend: "google" (requires internet, extremely fast) or "vinai" (offline Hugging Face model, requires RAM/GPU).
        """
        self.backend = backend.lower()
        self._hf_tokenizer = None
        self._hf_model = None
        self._hf_device = None
        self._google_translator = None
        
        if self.backend == "vinai":
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            print("Loading vinai/vinai-translate-vi2en...")
            self._hf_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._hf_tokenizer = AutoTokenizer.from_pretrained("vinai/vinai-translate-vi2en", src_lang="vi_VN", use_fast=False)
            self._hf_model = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-vi2en").to(self._hf_device)
        
        elif self.backend == "google":
            try:
                from deep_translator import GoogleTranslator
            except ImportError:
                raise ImportError("Please install deep-translator via: pip install deep-translator")
            self._google_translator = GoogleTranslator(source='vi', target='en')
            
        else:
            raise ValueError("Backend must be either 'google' or 'vinai'")

    def translate_term(self, term: str) -> str:
        """
        Translates a short clinical term from Vietnamese to English and cleans accidental grammar.
        """
        if not isinstance(term, str) or not term.strip():
            return ""
            
        term = term.strip().replace('_', ' ') # Fix for Vietnamese tokenized text
        
        try:
            if self.backend == "google":
                eng_term = self._google_translator.translate(term)
            elif self.backend == "vinai":
                input_ids = self._hf_tokenizer(term, return_tensors="pt").to(self._hf_device)
                output_ids = self._hf_model.generate(
                    **input_ids,
                    decoder_start_token_id=self._hf_tokenizer.lang_code_to_id["en_XX"],
                    num_return_sequences=1,
                    num_beams=5,
                    early_stopping=True
                )
                eng_term = self._hf_tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                
            return self._clean_term(eng_term)
        except Exception as e:
            print(f"Translation failed for '{term}': {e}")
            return term  # Fallback to original term if translation fails

    def translate_batch(self, terms: list, batch_size: int = 32, show_progress: bool = True) -> list:
        """
        Translates a list of terms simultaneously. 
        Massively faster for the 'vinai' backend by utilizing GPU batching.
        Automatically displays a tqdm progress bar unless show_progress is False.
        """
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = lambda x, **kwargs: x

        results = []
        
        # Google backend: Use deep-translator's native batching to avoid API rate limits
        if self.backend == "google":
            import time
            from tqdm import tqdm
            
            print(f"Translating {len(terms)} terms using Google API (Native Batching)...")
            
            # Google allows max 5k characters per request. Batching by 50 words is extremely safe.
            google_batch_size = 50 
            iterator = range(0, len(terms), google_batch_size)
            
            if show_progress:
                try:
                    iterator = tqdm(iterator, desc="Google Native Batch")
                except ImportError:
                    pass
                    
            for i in iterator:
                batch = [str(t).strip() for t in terms[i : i + google_batch_size]]
                try:
                    # Native deep-translator batch sends 1 HTTP request instead of 50!
                    eng_terms = self._google_translator.translate_batch(batch)
                    
                    # Clean terms just like VinAI does
                    for term in eng_terms:
                        results.append(self._clean_term(term))
                        
                except Exception as e:
                    if show_progress:
                        print(f"Batch failed: {e}. Falling back to original terms for this batch.")
                    results.extend(batch)
                    
                time.sleep(1) # Extra safety buffer for Google
                
            return results

        # VinAI backend uses GPU Tensor Batching
        if show_progress:
            print(f"Translating {len(terms)} terms using VinAI (Batch Size: {batch_size})...")
            
        iterator = range(0, len(terms), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="VinAI Batch Inference")
            
        for i in iterator:
            # CRITICAL FIX: Vietnamese NER tools use '_' to combine words (chóng_mặt).
            # Translation models expect natural spaces (chóng mặt). We must replace '_' with ' '.
            batch = [str(t).strip().replace('_', ' ') for t in terms[i : i + batch_size]]
            
            try:
                input_ids = self._hf_tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self._hf_device)
                output_ids = self._hf_model.generate(
                    **input_ids,
                    decoder_start_token_id=self._hf_tokenizer.lang_code_to_id["en_XX"],
                    num_return_sequences=1,
                    num_beams=5,
                    early_stopping=True
                )
                eng_terms = self._hf_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                results.extend([self._clean_term(t) for t in eng_terms])
            except Exception as e:
                if show_progress:
                    print(f"Batch failed: {e}. Falling back to original terms for this batch.")
                results.extend(batch)
                
        return results
        
    def _clean_term(self, text: str) -> str:
        """
        Cleans accidental grammar (like 'A', 'The') added by translation models to short nouns.
        """
        if not text:
            return ""
            
        text = text.strip().lower()
        
        # Remove common articles
        if text.startswith("a "):
            text = text[2:]
        elif text.startswith("an "):
            text = text[3:]
        elif text.startswith("the "):
            text = text[4:]
            
        # Remove trailing periods or punctuation that might have been added
        text = text.rstrip(".!?")
        
        return text.strip()

