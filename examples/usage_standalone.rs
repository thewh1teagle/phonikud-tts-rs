/*
Fully standalone interactive Hebrew TTS with embedded G2P and TTS models.
This embeds all models and configs directly into the binary.

Prepare models:
wget https://huggingface.co/thewh1teagle/phonikud-tts-checkpoints/resolve/main/shaul.onnx
wget https://huggingface.co/thewh1teagle/phonikud-tts-checkpoints/resolve/main/model.config.json
uv venv -p 3.12
uv pip install huggingface-hub
uv run hf download thewh1teagle/gemma3-270b-heb-g2p --local-dir gemma3_onnx

Build:
export SDKROOT=$(xcrun --show-sdk-path)
cargo build --release --example usage_standalone

Run:
./target/release/examples/usage_standalone

The compiled binary will be completely standalone and won't need any external files!
Binary size will be larger (~539MB) due to embedded models.

Features:
- Interactive loop: type text and press Enter to hear it spoken
- Completely standalone: no external files needed
- Embedded G2P (Gemma-based) and TTS (Piper-based) models
*/

use ort::{
    inputs,
    session::{Session, builder::GraphOptimizationLevel},
    value::{Value, TensorRef}
};
use ndarray::{Array1, Array2, Array4, Ix4};
use tokenizers::Tokenizer;
use anyhow::{Result, Context};
use serde::Deserialize;
use std::collections::HashMap;
use std::io::{stdin, stdout, Write, BufReader};
use std::fs::File;
use hound;
use rodio::{Decoder, Sink};

// Embed G2P model and tokenizer
const EMBEDDED_G2P_MODEL: &[u8] = include_bytes!("../../gemma3_onnx/model.onnx");
const EMBEDDED_G2P_TOKENIZER: &[u8] = include_bytes!("../../gemma3_onnx/tokenizer.json");

// Embed TTS model and config
const EMBEDDED_TTS_MODEL: &[u8] = include_bytes!("../../shaul.onnx");
const EMBEDDED_TTS_CONFIG: &str = include_str!("../../model.config.json");

// ==================== G2P Implementation ====================

struct G2pStandalone {
    session: Session,
    tokenizer: Tokenizer,
    num_layers: usize,
    num_key_value_heads: usize,
    head_dim: usize,
}

impl G2pStandalone {
    fn new() -> Self {
        println!("Loading G2P model from memory...");
        let session = Session::builder()
            .expect("Failed to create session builder")
            .with_optimization_level(GraphOptimizationLevel::Level1)
            .expect("Failed to set optimization level")
            .with_intra_threads(1)
            .expect("Failed to set intra threads")
            .commit_from_memory(EMBEDDED_G2P_MODEL)
            .expect("Failed to load G2P model from memory");
        
        let tokenizer = Tokenizer::from_bytes(EMBEDDED_G2P_TOKENIZER)
            .expect("Failed to load tokenizer from memory");
        
        let num_layers = 18;
        let num_key_value_heads = 1;
        let head_dim = 256;
        
        Self { 
            session, 
            tokenizer,
            num_layers,
            num_key_value_heads,
            head_dim,
        }
    }

    fn g2p(&mut self, text: &str) -> String {
        self.generate(text).expect("Failed to generate phonemes")
    }

    fn generate(&mut self, text: &str) -> Result<String> {
        let system_message = "Given the following Hebrew sentence, convert it to IPA phonemes.\nInput Format: A Hebrew sentence.\nOutput Format: A string of IPA phonemes.";
        
        let prompt = format!(
            "<start_of_turn>system\n{}<end_of_turn>\n<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n",
            system_message, text
        );

        let encoding = self.tokenizer
            .encode(prompt.clone(), false)
            .map_err(|e| anyhow::anyhow!("Failed to tokenize prompt: {}", e))?;
        
        let tokens_vec: Vec<i64> = encoding.get_ids().iter().map(|id| *id as i64).collect();
        
        let max_tokens = 150;
        let eos_token_id = 1i64;
        let end_of_turn_encoding = self.tokenizer.encode("<end_of_turn>", false).unwrap();
        let end_of_turn_token = end_of_turn_encoding.get_ids()[0] as i64;
        
        let mut generated_tokens = Vec::new();
        let mut past_key_values: Vec<Array4<f32>> = vec![
            Array4::zeros((1, self.num_key_value_heads, 0, self.head_dim)); 
            self.num_layers * 2
        ];
        
        let mut current_tokens = tokens_vec;
        let mut attention_mask_vec: Vec<i64> = vec![1; current_tokens.len()];
        
        for _ in 0..max_tokens {
            let seq_len = current_tokens.len() as i64;
            let input_ids = Array2::from_shape_vec((1, seq_len as usize), current_tokens.clone())?;
            
            let position_ids: Vec<i64> = if past_key_values[0].shape()[2] == 0 {
                (0..seq_len).collect()
            } else {
                vec![past_key_values[0].shape()[2] as i64]
            };
            let position_ids_array = Array2::from_shape_vec((1, position_ids.len()), position_ids)?;
            
            let attention_mask = Array2::from_shape_vec((1, attention_mask_vec.len()), attention_mask_vec.clone())?;
            
            let mut model_inputs = inputs![
                "input_ids" => TensorRef::from_array_view(&input_ids)?,
                "attention_mask" => TensorRef::from_array_view(&attention_mask)?,
                "position_ids" => TensorRef::from_array_view(&position_ids_array)?,
            ];
            
            for layer_idx in 0..self.num_layers {
                model_inputs.push((
                    format!("past_key_values.{}.key", layer_idx).into(),
                    TensorRef::from_array_view(&past_key_values[layer_idx * 2])?.into()
                ));
                model_inputs.push((
                    format!("past_key_values.{}.value", layer_idx).into(),
                    TensorRef::from_array_view(&past_key_values[layer_idx * 2 + 1])?.into()
                ));
            }
            
            let outputs = self.session.run(model_inputs)
                .context("Failed to run inference")?;
            
            let logits = outputs["logits"]
                .try_extract_array::<f32>()?
                .into_dimensionality::<ndarray::Ix3>()?;
            
            let last_token_logits = logits.slice(ndarray::s![0, -1, ..]);
            let next_token = last_token_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as i64)
                .context("Failed to sample next token")?;
            
            if next_token == eos_token_id || next_token == end_of_turn_token {
                break;
            }
            
            generated_tokens.push(next_token as u32);
            
            for layer_idx in 0..self.num_layers {
                past_key_values[layer_idx * 2] = outputs[format!("present.{}.key", layer_idx)]
                    .try_extract_array::<f32>()?
                    .into_dimensionality::<Ix4>()?
                    .to_owned();
                past_key_values[layer_idx * 2 + 1] = outputs[format!("present.{}.value", layer_idx)]
                    .try_extract_array::<f32>()?
                    .into_dimensionality::<Ix4>()?
                    .to_owned();
            }
            
            current_tokens = vec![next_token];
            attention_mask_vec.push(1);
        }
        
        let response = self.tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| anyhow::anyhow!("Failed to decode tokens: {}", e))?;
        
        let response = response
            .replace("<end_of_turn>", "")
            .replace("</s>", "")
            .replace("<eos>", "");
        
        Ok(response.trim().to_string())
    }
}

// ==================== TTS Implementation ====================

const BOS: &str = "^";
const EOS: &str = "$";
const PAD: &str = "_";

#[derive(Debug, Deserialize)]
struct Config {
    audio: AudioConfig,
    phoneme_id_map: HashMap<String, Vec<i64>>,
    inference: InferenceConfig,
}

#[derive(Debug, Deserialize)]
struct AudioConfig {
    sample_rate: i64,
}

#[derive(Debug, Deserialize)]
struct InferenceConfig {
    length_scale: f32,
    noise_scale: f32,
    noise_w: f32,
}

struct TtsStandalone {
    session: Session,
    config: Config,
    sample_rate: i64,
}

impl TtsStandalone {
    fn new() -> Result<Self> {
        println!("Loading TTS model from memory...");
        let config: Config = serde_json::from_str(EMBEDDED_TTS_CONFIG)?;
        let sample_rate = config.audio.sample_rate;

        let session = Session::builder()?
            .commit_from_memory(EMBEDDED_TTS_MODEL)?;

        Ok(Self {
            session,
            config,
            sample_rate,
        })
    }

    fn create(&mut self, phonemes: String) -> Result<(Vec<f32>, i64)> {
        let inference_cfg = &self.config.inference;
        
        let mut phoneme_chars: Vec<String> = vec![BOS.to_string()];
        phoneme_chars.extend(phonemes.chars().map(|c| c.to_string()));
        
        let ids = self.phoneme_to_ids(&phoneme_chars);
        
        let (input_ids, input_lengths, scales) = self.create_input(
            ids,
            inference_cfg.length_scale,
            inference_cfg.noise_w,
            inference_cfg.noise_scale,
        );

        let input_value = Value::from_array(input_ids)?;
        let lengths_value = Value::from_array(input_lengths)?;
        let scales_value = Value::from_array(scales)?;
        
        let outputs = self.session.run(inputs![
            "input" => &input_value,
            "input_lengths" => &lengths_value,
            "scales" => &scales_value,
        ])?;

        let output = outputs["output"].try_extract_tensor::<f32>()?;
        let samples: Vec<f32> = output.1.to_vec();

        Ok((samples, self.sample_rate))
    }

    fn phoneme_to_ids(&self, phonemes: &[String]) -> Vec<i64> {
        let mut ids = Vec::new();
        
        for phoneme in phonemes {
            if let Some(phoneme_ids) = self.config.phoneme_id_map.get(phoneme) {
                ids.extend(phoneme_ids);
                if let Some(pad_ids) = self.config.phoneme_id_map.get(PAD) {
                    ids.extend(pad_ids);
                }
            }
        }
        
        if let Some(eos_ids) = self.config.phoneme_id_map.get(EOS) {
            ids.extend(eos_ids);
        }
        
        ids
    }

    fn create_input(
        &self,
        ids: Vec<i64>,
        length_scale: f32,
        noise_w: f32,
        noise_scale: f32,
    ) -> (Array2<i64>, Array1<i64>, Array1<f32>) {
        let len = ids.len();
        let input_ids = Array2::from_shape_vec((1, len), ids).unwrap();
        let input_lengths = Array1::from_vec(vec![len as i64]);
        let scales = Array1::from_vec(vec![noise_scale, length_scale, noise_w]);
        
        (input_ids, input_lengths, scales)
    }
}

// ==================== Helper Functions ====================

fn save_wav(samples: &[f32], sample_rate: i64, path: &str) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sample_rate as u32,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(path, spec)?;

    for sample in samples {
        let amplitude = (sample * i16::MAX as f32) as i16;
        writer.write_sample(amplitude)?;
    }

    writer.finalize()?;
    Ok(())
}

fn play_audio(audio_path: &str) -> Result<()> {
    let (_stream, stream_handle) = rodio::OutputStream::try_default()?;
    let sink = Sink::try_new(&stream_handle)?;
    
    let file = BufReader::new(File::open(audio_path)?);
    let source = Decoder::new(file)?;
    
    sink.append(source);
    sink.sleep_until_end();
    Ok(())
}

// ==================== Main ====================

fn main() -> Result<()> {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║   Interactive Hebrew TTS - Standalone Edition             ║");
    println!("║   With Embedded G2P & TTS Models                          ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");
    
    println!("Loading embedded models...");
    let mut g2p = G2pStandalone::new();
    let mut tts = TtsStandalone::new()?;
    println!("✓ Models loaded successfully!\n");
    
    println!("Type Hebrew text and press Enter to hear it spoken.");
    println!("Type 'quit' or 'exit' to close the program.\n");
    
    loop {
        print!("Enter text: ");
        stdout().flush()?;
        
        let mut text = String::new();
        stdin().read_line(&mut text)?;
        let text = text.trim();
        
        if text.is_empty() {
            continue;
        }
        
        if text.eq_ignore_ascii_case("quit") || text.eq_ignore_ascii_case("exit") {
            println!("\n👋 Goodbye!");
            break;
        }
        
        println!("  → Converting to phonemes...");
        let phonemes = g2p.g2p(&text);
        println!("  → Phonemes: {}", phonemes);
        
        println!("  → Generating audio...");
        let (samples, sample_rate) = tts.create(phonemes)?;
        
        let output_path = "audio_temp.wav";
        save_wav(&samples, sample_rate, output_path)?;
        
        println!("  → Playing...");
        play_audio(output_path)?;
        println!();
    }
    
    Ok(())
}

