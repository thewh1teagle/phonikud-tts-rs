/*
wget https://huggingface.co/thewh1teagle/gemma3-heb-g2p-gguf/resolve/main/model.gguf
export SDKROOT=$(xcrun --show-sdk-path)
cargo run --example usage_g2p -- model.gguf
*/
use phonikud_tts_rs::g2p::G2p;

fn main() {
    let g2p_model_path = std::env::args()
        .nth(1)
        .expect("Please specify model path: cargo run --example usage_g2p -- model.gguf");
    
    let text = "שלום עולם! מה קורה?";
    let g2p = G2p::new(g2p_model_path);
    let phonemes = g2p.g2p(text);
    println!("Phonemes: {}", phonemes);
}