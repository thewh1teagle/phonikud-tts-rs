/*
Standalone example with embedded model and tokenizer.
This embeds the ONNX model and tokenizer directly into the binary.

Build:
export SDKROOT=$(xcrun --show-sdk-path)
cargo build --release --example usage_g2p_standalone

Run:
./target/release/examples/usage_g2p_standalone

The compiled binary will be completely standalone and won't need external model files!
*/

use ort::{
    inputs,
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef
};
use ndarray::{Array2, Array4, Ix4};
use tokenizers::Tokenizer;
use anyhow::{Result, Context};

// Embed the model and tokenizer directly into the binary
const EMBEDDED_MODEL: &[u8] = include_bytes!("../../gemma3_onnx/model.onnx");
const EMBEDDED_TOKENIZER: &[u8] = include_bytes!("../../gemma3_onnx/tokenizer.json");

struct G2pStandalone {
    session: Session,
    tokenizer: Tokenizer,
    num_layers: usize,
    num_key_value_heads: usize,
    head_dim: usize,
}

impl G2pStandalone {
    fn new() -> Self {
        // Load ONNX model from embedded bytes
        let session = Session::builder()
            .expect("Failed to create session builder")
            .with_optimization_level(GraphOptimizationLevel::Level1)
            .expect("Failed to set optimization level")
            .with_intra_threads(1)
            .expect("Failed to set intra threads")
            .commit_from_memory(EMBEDDED_MODEL)
            .expect("Failed to load model from memory");
        
        // Load tokenizer from embedded bytes
        let tokenizer = Tokenizer::from_bytes(EMBEDDED_TOKENIZER)
            .expect("Failed to load tokenizer from memory");
        
        // Model config from config.json
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
        
        // Format the prompt using Gemma chat template
        let prompt = format!(
            "<start_of_turn>system\n{}<end_of_turn>\n<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n",
            system_message, text
        );

        // Tokenize the prompt
        let encoding = self.tokenizer
            .encode(prompt.clone(), false)
            .map_err(|e| anyhow::anyhow!("Failed to tokenize prompt: {}", e))?;
        
        let tokens_vec: Vec<i64> = encoding.get_ids().iter().map(|id| *id as i64).collect();
        
        let max_tokens = 150;
        let eos_token_id = 1i64;
        let end_of_turn_encoding = self.tokenizer.encode("<end_of_turn>", false).unwrap();
        let end_of_turn_token = end_of_turn_encoding.get_ids()[0] as i64;
        
        let mut generated_tokens = Vec::new();
        
        // Initialize past_key_values for the transformer model
        let mut past_key_values: Vec<Array4<f32>> = vec![
            Array4::zeros((1, self.num_key_value_heads, 0, self.head_dim)); 
            self.num_layers * 2
        ];
        
        let mut current_tokens = tokens_vec;
        let mut attention_mask_vec: Vec<i64> = vec![1; current_tokens.len()];
        
        for _ in 0..max_tokens {
            // Prepare input tensor [batch_size, sequence_length]
            let seq_len = current_tokens.len() as i64;
            let input_ids = Array2::from_shape_vec((1, seq_len as usize), current_tokens.clone())?;
            
            // Prepare position_ids tensor [batch_size, sequence_length]
            let position_ids: Vec<i64> = if past_key_values[0].shape()[2] == 0 {
                (0..seq_len).collect()
            } else {
                vec![past_key_values[0].shape()[2] as i64]
            };
            let position_ids_array = Array2::from_shape_vec((1, position_ids.len()), position_ids)?;
            
            // Prepare attention mask
            let attention_mask = Array2::from_shape_vec((1, attention_mask_vec.len()), attention_mask_vec.clone())?;
            
            // Prepare model inputs using the inputs! macro
            let mut model_inputs = inputs![
                "input_ids" => TensorRef::from_array_view(&input_ids)?,
                "attention_mask" => TensorRef::from_array_view(&attention_mask)?,
                "position_ids" => TensorRef::from_array_view(&position_ids_array)?,
            ];
            
            // Add KV cache
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
            
            // Run inference
            let outputs = self.session.run(model_inputs)
                .context("Failed to run inference")?;
            
            // Get logits from output
            let logits = outputs["logits"]
                .try_extract_array::<f32>()?
                .into_dimensionality::<ndarray::Ix3>()?;
            
            // Get the last token's logits and find the token with highest probability (greedy sampling)
            let last_token_logits = logits.slice(ndarray::s![0, -1, ..]);
            let next_token = last_token_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as i64)
                .context("Failed to sample next token")?;
            
            // Check for end of generation
            if next_token == eos_token_id || next_token == end_of_turn_token {
                break;
            }
            
            generated_tokens.push(next_token as u32);
            
            // Update KV cache from outputs
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
            
            // For next iteration, only use the newly generated token
            current_tokens = vec![next_token];
            // Extend attention mask for the new token
            attention_mask_vec.push(1);
        }
        
        // Decode the generated tokens
        let response = self.tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| anyhow::anyhow!("Failed to decode tokens: {}", e))?;
        
        // Clean up any remaining special tokens
        let response = response
            .replace("<end_of_turn>", "")
            .replace("</s>", "")
            .replace("<eos>", "");
        
        Ok(response.trim().to_string())
    }
}

fn main() {
    println!("Loading embedded model...");
    let mut g2p = G2pStandalone::new();
    
    let text = "שלום עולם! מה קורה?";
    println!("Input: {}", text);
    
    let phonemes = g2p.g2p(text);
    println!("Phonemes: {}", phonemes);
    
    // Try another example
    let text2 = "תודה רבה";
    println!("\nInput: {}", text2);
    let phonemes2 = g2p.g2p(text2);
    println!("Phonemes: {}", phonemes2);
}

