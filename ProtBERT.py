import torch
from transformers import T5Tokenizer, T5EncoderModel
import pandas as pd
from tqdm import tqdm

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_bfd", do_lower_case=False)
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_bfd")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

# Load FASTA and parse headers + sequences
def load_fasta(file_path):
    sequences = []
    headers = []
    with open(file_path, 'r') as f:
        current_seq = ''
        current_header = ''
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    sequences.append(current_seq.replace(" ", ""))
                    headers.append(current_header)
                    current_seq = ''
                current_header = line[1:]  # remove '>'
            else:
                current_seq += line
        # append last
        if current_seq:
            sequences.append(current_seq.replace(" ", ""))
            headers.append(current_header)
    return headers, sequences

# Convert protein sequence to embedding
def embed_sequence(sequence):
    spaced_seq = ' '.join(list(sequence))
    tokens = tokenizer(spaced_seq, return_tensors='pt', padding=True).input_ids.to(device)
    with torch.no_grad():
        embedding = model(input_ids=tokens).last_hidden_state
    # Remove special tokens, mean pooling
    embedding = embedding[0][1:-1]
    return torch.mean(embedding, dim=0).cpu().numpy()

# Main process
def fasta_to_embeddings(input_fasta_file, output_csv_file):
    headers, sequences = load_fasta(input_fasta_file)
    embeddings = []
    ids = []

    for header, seq in tqdm(zip(headers, sequences), total=len(sequences), desc="Extracting features"):
        try:
            emb = embed_sequence(seq)
            embeddings.append(emb)
            ids.append(header.split()[0])  # Use first token in header as ID
        except Exception as e:
            print(f"Error processing {header}: {e}")

    # Create DataFrame with proper column names
    df = pd.DataFrame(embeddings)
    # Rename columns to ProtT5-BFD_F0, ProtT5-BFD_F1, etc.
    df.columns = [f"ProtT5-BFD_F{i+1}" for i in range(df.shape[1])]
    # Insert Sequence_ID as first column
    df.insert(0, "Sequence_ID", ids)
    
    df.to_csv(output_csv_file, index=False)
    print(f"Saved embeddings to {output_csv_file}")

# Example usage
input_file = "/kaggle/input/gb-test-198-182-fasta/GB_test_198_182.fasta"  # Replace with your FASTA file
output_file = "ProtT5_BFD_embeddings_test.csv" #give name to your output file 
fasta_to_embeddings(input_file, output_file)