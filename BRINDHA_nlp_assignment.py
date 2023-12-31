#installing requirements
! pip3 install transformers
! pip3 install datasets
! pip install accelerate
! pip install --upgrade tensorflow

##using predefined model
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
text = "hello guys"
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
print(output)


#using whisper -large -v3 model
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]

result = pipe(sample)
print(result["text"])


#adding marathi audio data
sample = '/content/common_voice_mr_31917739.wav'
result = pipe(sample)
print(result["text"])

##adding text for the marathi audio data
reference = 'त्यानुसार कृषी विद्यापीठातील शिक्षणक्रम राबविले जातात'


def calculate_wer(reference, hypothesis):
    ref_chars = list(reference)  
    hyp_chars = list(hypothesis) 

    
    substitutions = sum(                             
    1 for ref, hyp in zip(ref_chars, hyp_chars)  
    if ref != hyp                               
    )

    deletions = max(len(ref_chars) - len(hyp_chars), 0)
    insertions = max(len(hyp_chars) - len(ref_chars), 0)

    total_chars = max(len(ref_chars), 1)  # Avoid division by zero
    cer = (substitutions + deletions + insertions) / total_chars
    return cer


#Hyperparameter tuning:
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset

# Hyperparameters for tuning
max_new_tokens_values = [64, 128, 256] 
chunk_length_s_values = [20, 30, 40]  

best_wer = float('inf')  
best_hyperparams = {}

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

for max_tokens in max_new_tokens_values:
    for chunk_length in chunk_length_s_values:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=max_tokens,
            chunk_length_s=chunk_length,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=torch_dtype,
            device=device,
        )

        dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
        sample = dataset[0]["audio"]

        result = pipe(sample)
        print(f"WER for max_tokens={max_tokens}, chunk_length={chunk_length}: {calculate_wer(reference, result['text'])}")

        current_wer = calculate_wer(reference, result["text"])
        if current_wer < best_wer:
            best_wer = current_wer
            best_hyperparams = {'max_tokens': max_tokens, 'chunk_length': chunk_length}

print(f"Best hyperparameters: {best_hyperparams}, Best WER: {best_wer}")


# Calculating Character Error Rate (CER)
def calculate_cer(reference, hypothesis):
    ref_chars = list(reference)  
    hyp_chars = list(hypothesis) 

    # Counting the number of substitutions, deletions, and insertions
    substitutions = sum(1 for ref, 
                        hyp in zip(ref_chars, 
                        hyp_chars) if ref != hyp
                        )
    deletions = max(len(ref_chars) - len(hyp_chars), 0)
    insertions = max(len(hyp_chars) - len(ref_chars), 0)

    total_chars = max(len(ref_chars), 1)  # Avoid division by zero
    cer = (substitutions + deletions + insertions) / total_chars
    return cer

# Calculating Sentence Error Rate (SER)
def calculate_ser(reference_sentences, hypothesis_sentences):
    total_sentences = len(reference_sentences)
    if total_sentences != len(hypothesis_sentences):
        raise ValueError("Number of reference and hypothesis sentences should match.")

    # Calculating CER for each sentence
    error_sentences = sum(1 for ref, hyp in zip(reference_sentences, hypothesis_sentences)
                          if calculate_cer(ref, hyp) > threshold)

    # Calculating SER for each sentence
    ser = error_sentences / total_sentences if total_sentences > 0 else 0
    return ser

reference_sentences = ['त्यानुसार कृषी विद्यापीठातील शिक्षणक्रम राबविले जातात']
hypothesis_sentences = ['त्या नुसार कृषिव विद्यापितातील शिक्षन क्रम राभवले जातात।',]
threshold = 0.1  

# for CER
cer = calculate_cer(reference_sentences[0], hypothesis_sentences[0])
print(f"Character Error Rate (CER): {cer}")

# for SER
ser = calculate_ser(reference_sentences, hypothesis_sentences)
print(f"Sentence Error Rate (SER): {ser}")
