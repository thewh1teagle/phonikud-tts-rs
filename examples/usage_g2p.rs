/*
Usage:
export SDKROOT=$(xcrun --show-sdk-path)
cargo run --example usage_g2p -- ../gemma3_onnx

Or specify a custom path:
cargo run --example usage_g2p -- /path/to/gemma3_onnx
*/
use phonikud_tts_rs::g2p::G2p;

fn main() {
    let g2p_model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "../gemma3_onnx".to_string());
    
    println!("Loading model from: {}", g2p_model_path);
    let mut g2p = G2p::new(g2p_model_path);
    
    let text = "שלום עולם! מה קורה?";
    println!("Input: {}", text);
    
    let phonemes = g2p.g2p(text);
    println!("Phonemes: {}", phonemes);
}