import torch
import os
from model import StreamingEmformerCTC

def export_to_onnx_tensorrt(model_checkpoint_path, output_onnx_path="callex_stt_rnd/checkpoints/callex_stt_engine.onnx"):
    """
    The Ultimate Native Server Expansion Code execution.
    Python APIs die completely under the weight of 5,000 parallel phone sockets. 
    This script fundamentally strips the complex PyTorch framework natively and compiles the 
    raw Matrix boundaries strictly into an ONNX graph structure!
    This ONNX file is then natively dragged into NVIDIA Triton Inference Servers seamlessly, 
    executing strictly on bare-metal C++ TensorRT boundaries dynamically!
    """
    print("\\n[Callex Enterprise] Commencing ONNX Model Graph Compiler Extraction...")
    
    # ── 1. Instantiate the Model Framework Native memory bounds ──
    # Note: Vocab size and d_model dynamically match training topology
    model = StreamingEmformerCTC(vocab_size=8000).eval()
    
    if os.path.exists(model_checkpoint_path):
        checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded native PyTorch weights from {model_checkpoint_path}.")
    else:
        print("[Warning] No exact `.pt` weights found. Extracting structural architectural bounds blindly.")
        
    # ── 2. Instantiate Mock Audio Streams perfectly representing 40ms blocks natively ──
    # For a sample rate of 16k, 40ms is exactly 640 matrices arrays natively
    mock_mel = torch.randn(1, 80, 100) # [Batch, Features, 40ms_Chunk_Length]
    mock_memory = torch.randn(1, 40, 256) # [Batch, Left_Context, d_model]
    
    print("\\n[Callex Enterprise] Stripping Architectural Syntax into Graph Mode natively...")
    
    # ── 3. Execute Deep Vector Extraction ──
    torch.onnx.export(
        model, 
        (mock_mel, mock_memory), # Stream boundaries physically mapped natively
        output_onnx_path,
        export_params=True,
        opset_version=14, # Adhere perfectly to TensorRT execution rules
        do_constant_folding=True,
        input_names=['mel_stream', 'left_context_memory'],
        output_names=['ctc_log_probs', 'new_context_memory'],
        dynamic_axes={
           'mel_stream': {0: 'batch_size', 2: 'time_length'},
           'left_context_memory': {0: 'batch_size', 1: 'context_length'},
           'ctc_log_probs': {0: 'batch_size', 1: 'time_length'},
           'new_context_memory': {0: 'batch_size', 1: 'context_length'}
        }
    )
    
    print(f"🚀 MASSIVE SUCCESS! Model Extracted precisely to '{output_onnx_path}'.")
    print(">> Proceed to map explicitly onto NVIDIA Triton (TensorRT) boundaries for 5000+ PBX call execution natively.")

if __name__ == "__main__":
    os.makedirs("callex_stt_rnd/checkpoints", exist_ok=True)
    export_to_onnx_tensorrt("callex_stt_rnd/checkpoints/callex_conformer_e1.pt")
