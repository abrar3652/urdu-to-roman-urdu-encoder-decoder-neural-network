import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
import random
import math

# Model Classes (unchanged)
class BiLSTMEncoder(nn.Module):
    """BiLSTM Encoder (2 layers as required)"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, dropout=0.1):
        super(BiLSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers, 
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, lengths=None):
        batch_size = x.size(0)
        embedded = self.dropout(self.embedding(x))
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )
            outputs, (hidden, cell) = self.lstm(packed)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        else:
            outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell

class Attention(nn.Module):
    """Attention mechanism with masking support"""
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, hidden, encoder_outputs, mask=None):
        batch_size, seq_len, _ = encoder_outputs.size()
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.cat([hidden, encoder_outputs], dim=2)
        energy = torch.tanh(self.attn(energy))
        attention = self.v(energy).squeeze(2)
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)
        attention_weights = F.softmax(attention, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)
        return context, attention_weights

class LSTMDecoder(nn.Module):
    """LSTM Decoder with attention (4 layers as required)"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=4, dropout=0.1):
        super(LSTMDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = Attention(hidden_dim)
        self.lstm = nn.LSTM(
            embed_dim + hidden_dim * 2, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.output_projection = nn.Linear(hidden_dim + hidden_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, hidden_state, encoder_outputs, encoder_mask=None):
        if x.dim() == 1:
            x = x.unsqueeze(1)
        embedded = self.dropout(self.embedding(x))
        context, attention_weights = self.attention(
            hidden_state[0][-1], encoder_outputs, encoder_mask
        )
        lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        output, hidden_state = self.lstm(lstm_input, hidden_state)
        final_output = torch.cat([output, context.unsqueeze(1)], dim=2)
        predictions = self.output_projection(final_output)
        return predictions, hidden_state, attention_weights

class Seq2SeqModel(nn.Module):
    """Seq2Seq model with BiLSTM encoder and LSTM decoder"""
    def __init__(self, urdu_vocab_size, roman_vocab_size, embed_dim=128, hidden_dim=256,
                 encoder_layers=2, decoder_layers=4, dropout=0.1):
        super(Seq2SeqModel, self).__init__()
        self.encoder = BiLSTMEncoder(urdu_vocab_size, embed_dim, hidden_dim, encoder_layers, dropout)
        self.decoder = LSTMDecoder(roman_vocab_size, embed_dim, hidden_dim, decoder_layers, dropout)
        self.hidden_dim = hidden_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.bridge_h = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bridge_c = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, urdu_seq, roman_seq=None, teacher_forcing_ratio=0.5):
        batch_size = urdu_seq.size(0)
        device = urdu_seq.device
        encoder_mask = (urdu_seq != 0).float()
        encoder_lengths = encoder_mask.sum(dim=1).cpu()
        encoder_outputs, encoder_hidden, encoder_cell = self.encoder(urdu_seq, encoder_lengths)
        encoder_hidden = encoder_hidden.view(self.encoder_layers, 2, batch_size, self.hidden_dim)
        encoder_cell = encoder_cell.view(self.encoder_layers, 2, batch_size, self.hidden_dim)
        last_hidden = torch.cat([encoder_hidden[-1, 0], encoder_hidden[-1, 1]], dim=1)
        last_cell = torch.cat([encoder_cell[-1, 0], encoder_cell[-1, 1]], dim=1)
        decoder_hidden = self.bridge_h(last_hidden).unsqueeze(0).repeat(self.decoder_layers, 1, 1)
        decoder_cell = self.bridge_c(last_cell).unsqueeze(0).repeat(self.decoder_layers, 1, 1)
        
        if roman_seq is not None:
            max_length = roman_seq.size(1)
            outputs = []
            input_token = roman_seq[:, 0:1]
            hidden_state = (decoder_hidden, decoder_cell)
            for t in range(max_length - 1):
                output, hidden_state, _ = self.decoder(
                    input_token, hidden_state, encoder_outputs, encoder_mask
                )
                outputs.append(output)
                if random.random() < teacher_forcing_ratio:
                    input_token = roman_seq[:, t+1:t+2]
                else:
                    input_token = output.argmax(dim=-1)
            return torch.cat(outputs, dim=1)
        else:
            return self.beam_search_decode(
                encoder_outputs, encoder_mask, decoder_hidden, decoder_cell, 
                beam_size=3, max_length=50
            )
    
    def beam_search_decode(self, encoder_outputs, encoder_mask, decoder_hidden, 
                          decoder_cell, beam_size=3, max_length=50):
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        if batch_size > 1:
            return self.greedy_decode(encoder_outputs, encoder_mask, decoder_hidden, 
                                     decoder_cell, max_length)
        beams = [([], 0.0, (decoder_hidden, decoder_cell))]
        completed = []
        start_token = torch.tensor([[2]], dtype=torch.long, device=device)
        for step in range(max_length):
            candidates = []
            for seq, score, hidden_state in beams:
                if len(seq) > 0 and seq[-1] == 3:
                    completed.append((seq, score))
                    continue
                input_token = start_token if len(seq) == 0 else torch.tensor([[seq[-1]]], dtype=torch.long, device=device)
                output, new_hidden, _ = self.decoder(
                    input_token, hidden_state, encoder_outputs, encoder_mask
                )
                log_probs = F.log_softmax(output.squeeze(1), dim=-1)
                top_k_scores, top_k_tokens = log_probs.topk(beam_size)
                for k in range(beam_size):
                    token = top_k_tokens[0, k].item()
                    token_score = top_k_scores[0, k].item()
                    if len(seq) > 0 and token == seq[-1]:
                        token_score -= 2.0
                    if len(seq) > 3 and seq[-3:].count(token) > 1:
                        token_score -= 3.0
                    new_seq = seq + [token]
                    new_score = score + token_score
                    candidates.append((new_seq, new_score, new_hidden))
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_size]
            if len(beams) == 0:
                break
        completed.extend(beams)
        best_seq = max(completed, key=lambda x: x[1] / len(x[0]))[0] if completed else beams[0][0] if beams else []
        output_tensor = torch.tensor([best_seq], dtype=torch.long, device=device) if best_seq else torch.tensor([[3]], dtype=torch.long, device=device)
        vocab_size = self.decoder.vocab_size
        output_probs = torch.zeros(1, len(best_seq), vocab_size, device=device)
        for i, token in enumerate(best_seq):
            output_probs[0, i, token] = 1.0
        return output_probs
    
    def greedy_decode(self, encoder_outputs, encoder_mask, decoder_hidden, 
                     decoder_cell, max_length=50):
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        vocab_size = self.decoder.vocab_size
        outputs = []
        input_token = torch.full((batch_size, 1), 2, dtype=torch.long, device=device)
        hidden_state = (decoder_hidden, decoder_cell)
        recent_tokens = []
        for step in range(max_length):
            output, hidden_state, _ = self.decoder(
                input_token, hidden_state, encoder_outputs, encoder_mask
            )
            if len(recent_tokens) > 0:
                for recent_token in recent_tokens[-3:]:
                    output[0, 0, recent_token] -= 5.0
            outputs.append(output)
            input_token = output.argmax(dim=-1)
            token_id = input_token.item() if batch_size == 1 else input_token[0].item()
            recent_tokens.append(token_id)
            if token_id == 3:
                break
        return torch.cat(outputs, dim=1)

# Inference Function (unchanged)
def translate_text(model, text, urdu_tokenizer, roman_tokenizer):
    """Translate a single text"""
    model.eval()
    if not text.strip():
        return "Error: Input text is empty"
    try:
        tokens = urdu_tokenizer.encode(text, out_type=int)
        input_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            predicted_tokens = output.argmax(dim=-1).squeeze().cpu().tolist()
            try:
                eos_index = predicted_tokens.index(3)
                predicted_tokens = predicted_tokens[:eos_index]
            except ValueError:
                pass
            clean_tokens = [t for t in predicted_tokens if t not in [0, 1, 2]]
            translated_text = roman_tokenizer.decode(clean_tokens)
        return translated_text
    except Exception as e:
        return f"Error during translation: {str(e)}"

# Set device to CPU
device = torch.device('cpu')

# Load tokenizers
try:
    urdu_tokenizer = spm.SentencePieceProcessor(model_file='urdu_tokenizer.model')
    roman_tokenizer = spm.SentencePieceProcessor(model_file='roman_tokenizer.model')
except FileNotFoundError as e:
    st.error(f"Tokenizer file missing: {e}. Ensure 'urdu_tokenizer.model' and 'roman_tokenizer.model' are in the directory.")
    st.stop()

# Get vocab sizes
urdu_vocab_size = urdu_tokenizer.get_piece_size()
roman_vocab_size = roman_tokenizer.get_piece_size()

# Initialize and load model
try:
    model = Seq2SeqModel(
        urdu_vocab_size=urdu_vocab_size,
        roman_vocab_size=roman_vocab_size,
        embed_dim=128,
        hidden_dim=256,
        dropout=0.1
    ).to(device)
    model.load_state_dict(torch.load('best_model_baseline.pth', map_location=device))
    model.eval()
except FileNotFoundError as e:
    st.error(f"Model file missing: {e}. Ensure 'best_model_baseline.pth' is in the directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Streamlit UI with Enhanced Features
# Custom CSS for styling and animations
st.markdown("""
<style>
/* General styling */
body {
    font-family: 'Arial', sans-serif;
}

/* Naqsh-e-Urdu title with gradient and typing animation */
.naqsh-title {
    font-size: 3em;
    font-weight: bold;
    background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
    -webkit-background-clip: text;
    color: transparent;
    display: inline-block;
    animation: typing 2s steps(20, end), blink-caret .5s step-end infinite alternate;
    white-space: nowrap;
    overflow: hidden;
    border-right: 3px solid #333;
}
@keyframes typing {
    from { width: 0; }
    to { width: 100%; }
}
@keyframes blink-caret {
    from, to { border-color: transparent; }
    50% { border-color: #333; }
}

/* Fade-in animation for results */
.fade-in {
    animation: fadeIn 0.5s ease-in;
}
@keyframes fadeIn {
    0% { opacity: 0; transform: translateY(10px); }
    100% { opacity: 1; transform: translateY(0); }
}

/* Button styling with hover effect */
.stButton>button {
    background-color: #4ecdc4;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background-color: #ff6b6b;
    transform: scale(1.05);
}

/* Clear button styling */
.clear-button>button {
    background-color: #ff6b6b;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
    transition: all 0.3s ease;
}
.clear-button>button:hover {
    background-color: #e55a5a;
    transform: scale(1.05);
}

/* Copy button styling */
.copy-button>button {
    background-color: #45b7d1;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
    transition: all 0.3s ease;
}
.copy-button>button:hover {
    background-color: #3a9bb1;
    transform: scale(1.05);
}

/* Text area styling */
.stTextArea textarea {
    border: 2px solid #4ecdc4;
    border-radius: 5px;
    padding: 10px;
    font-size: 1.1em;
}
.stTextArea textarea:focus {
    border-color: #ff6b6b;
    box-shadow: 0 0 5px rgba(255, 107, 107, 0.5);
}
</style>
""", unsafe_allow_html=True)

# Model name with animated title
st.markdown('<div class="naqsh-title">Naqsh-e-Urdu</div>', unsafe_allow_html=True)
st.markdown("Enter Urdu text below to get its Roman Urdu translation.")

# Initialize session state for input and output
if 'urdu_input' not in st.session_state:
    st.session_state.urdu_input = ""
if 'result' not in st.session_state:
    st.session_state.result = ""

# Text area with unique key
input_key = "urdu_input_area"
urdu_input = st.text_area("Urdu Text:", value=st.session_state.urdu_input, 
                         placeholder="Enter Urdu text here (e.g., دل کی بات)", 
                         key=input_key)

# JavaScript for Enter key detection
st.markdown("""
<script>
document.addEventListener("DOMContentLoaded", function() {
    const textarea = document.querySelector("textarea");
    textarea.addEventListener("keydown", function(event) {
        if (event.key === "Enter" && !event.shiftKey && !event.ctrlKey) {
            event.preventDefault();
            document.querySelectorAll("p > button")[0].click();  // Targets the first button (Translate)
        }
    });
});
</script>
""", unsafe_allow_html=True)

# Buttons in a row
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button("Translate"):
        if urdu_input.strip():
            with st.spinner("Translating..."):
                result = translate_text(model, urdu_input, urdu_tokenizer, roman_tokenizer)
                st.session_state.urdu_input = urdu_input
                st.session_state.result = result
        else:
            st.session_state.result = "Error: Input text is empty"
with col2:
    if st.button("Clear", key="clear_button", help="Clear the input text"):
        st.session_state.urdu_input = ""
        st.session_state.result = ""
        st.rerun()
with col3:
    if st.session_state.result and not st.session_state.result.startswith("Error"):
        if st.button("Copy Result", key="copy_button"):
            st.markdown(f"""
            <script>
            navigator.clipboard.writeText("{st.session_state.result}");
            alert("Result copied to clipboard!");
            </script>
            """, unsafe_allow_html=True)

# Display result
if st.session_state.result:
    if st.session_state.result.startswith("Error"):
        st.markdown(f'<div class="fade-in"><span style="color: red;">{st.session_state.result}</span></div>', 
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="fade-in"><b>Roman Urdu Translation:</b> {st.session_state.result}</div>', 
                    unsafe_allow_html=True)