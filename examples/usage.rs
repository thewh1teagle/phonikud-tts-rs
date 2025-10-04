/*
wget https://huggingface.co/thewh1teagle/phonikud-tts-checkpoints/resolve/main/shaul.onnx
wget https://huggingface.co/thewh1teagle/phonikud-tts-checkpoints/resolve/main/model.config.json
wget https://huggingface.co/thewh1teagle/gemma3-heb-g2p-gguf/resolve/main/model.gguf

export SDKROOT=$(xcrun --show-sdk-path)
cargo run --example usage -- shaul.onnx model.config.json model.gguf
*/

use phonikud_tts_rs::tts::Tts;
use phonikud_tts_rs::g2p::G2p;
use phonikud_tts_rs::save_wav;
use anyhow::Result;

fn main() -> Result<()> {
    let tts_model_path = std::env::args()
        .nth(1)
        .expect("Please specify model path: cargo run --example usage -- shaul.onnx model.config.json model.gguf");
    let tts_config_path = std::env::args()
        .nth(2)
        .expect("Please specify config path: cargo run --example usage -- shaul.onnx model.config.json model.gguf");
    let g2p_model_path = std::env::args()
        .nth(3)
        .expect("Please specify g2p model path: cargo run --example usage -- shaul.onnx model.config.json model.gguf");

    let text = "שלום עולם! מה קורה? האם תצליח לדבר בעברית?";

    let mut tts = Tts::new(tts_model_path, tts_config_path)?;
    let mut g2p = G2p::new(g2p_model_path);
    let phonemes = g2p.g2p(&text);
    println!("Phonemes: {}", phonemes);
    let (samples, sample_rate) = tts.create(phonemes)?;
    println!("Samples: {}", samples.len());
    println!("Sample rate: {}", sample_rate);
    let output_path = "audio.wav";
    save_wav(&samples, sample_rate, output_path)?;
    println!("Audio saved to {}", output_path);
    Ok(())
}