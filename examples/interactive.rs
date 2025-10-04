/*
wget https://huggingface.co/thewh1teagle/phonikud-tts-checkpoints/resolve/main/shaul.onnx
wget https://huggingface.co/thewh1teagle/phonikud-tts-checkpoints/resolve/main/model.config.json
wget https://huggingface.co/thewh1teagle/gemma3-heb-g2p-gguf/resolve/main/model.gguf

export SDKROOT=$(xcrun --show-sdk-path)
cargo build --example interactive
./target/debug/examples/interactive
*/

use phonikud_tts_rs::tts::Tts;
use phonikud_tts_rs::g2p::G2p;
use phonikud_tts_rs::save_wav;
use anyhow::Result;
use rodio::{Decoder, Sink};
use std::io::{stdin, stdout, Write, BufReader};
use std::fs::File;


fn play_audio(audio_path: &str) {
    let (_stream, stream_handle) = rodio::OutputStream::try_default().unwrap();
    let sink = Sink::try_new(&stream_handle).unwrap();
    
    let file = BufReader::new(File::open(audio_path).unwrap());
    let source = Decoder::new(file).unwrap();
    
    sink.append(source);
    sink.sleep_until_end();
}

fn main() -> Result<()> {
    println!("Loading models from embedded bytes...");
    let mut tts = Tts::new("model.onnx".to_string(), "model.config.json".to_string()    )?;
    let mut g2p = G2p::new("model.gguf".to_string());
    println!("Models loaded successfully!");
    println!();
    
    loop {
        print!("Type your text (or 'quit' to exit): ");
        stdout().flush()?;
        
        let mut text = String::new();
        stdin().read_line(&mut text)?;
        let text = text.trim();
        
        if text.is_empty() {
            continue;
        }
        
        if text.eq_ignore_ascii_case("quit") || text.eq_ignore_ascii_case("exit") {
            println!("Goodbye!");
            break;
        }
        
        println!("Converting text to phonemes...");
        let phonemes = g2p.g2p(&text);
        println!("Phonemes: {}", phonemes);
        
        println!("Generating audio...");
        let (samples, sample_rate) = tts.create(phonemes)?;
        
        let output_path = "audio.wav";
        save_wav(&samples, sample_rate, output_path)?;
        println!("Audio saved to {}", output_path);
        
        println!("Playing audio...");
        play_audio(output_path);
        println!();
    }
    
    Ok(())
}