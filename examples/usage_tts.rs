/*
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json
cargo run --example usage
*/
use phonikud_tts_rs::tts::Tts;
use phonikud_tts_rs::save_wav;
use anyhow::Result;

fn main() -> Result<()> {
    let mut tts = Tts::new(
        "en_US-ryan-medium.onnx".to_string(),
        "en_US-ryan-medium.onnx.json".to_string(),
    )?;
    
    let phonemes = "wˈɪzdəm ɪz ðə ɹəwˈɔɹd ju ɡɛt fɔɹ ɐ lˈIftˌIm ʌv lˈɪsənɪŋ wˌɛn jud hæv pɹifˈɜɹd tə tˈɔk.";
    
    println!("Generating audio from phonemes: {}", phonemes);
    let (samples, sample_rate) = tts.create(phonemes.to_string())?;
    
    println!("Generated {} samples at {} Hz", samples.len(), sample_rate);
    
    let output_path = "audio.wav";
    save_wav(&samples, sample_rate, output_path)?;
    
    println!("Audio saved to {}", output_path);
    
    Ok(())
}