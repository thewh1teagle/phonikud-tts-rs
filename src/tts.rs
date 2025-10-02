use anyhow::Result;
use ort::{session::Session, value::Value, inputs};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use ndarray::{Array1, Array2};

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

pub struct Tts {
    session: Session,
    config: Config,
    sample_rate: i64,
}

impl Tts {
    pub fn new(tts_model_path: String, tts_config_path: String) -> Result<Self> {
        // Load config
        let config_content = fs::read_to_string(&tts_config_path)?;
        let config: Config = serde_json::from_str(&config_content)?;
        let sample_rate = config.audio.sample_rate;

        // Load ONNX model
        let session = Session::builder()?
            .commit_from_file(&tts_model_path)?;

        Ok(Self {
            session,
            config,
            sample_rate,
        })
    }

    pub fn create(&mut self, phonemes: String) -> Result<(Vec<f32>, i64)> {
        let inference_cfg = &self.config.inference;
        
        // Process phonemes
        let mut phoneme_chars: Vec<String> = vec![BOS.to_string()];
        phoneme_chars.extend(phonemes.chars().map(|c| c.to_string()));
        
        // Convert phonemes to IDs
        let ids = self.phoneme_to_ids(&phoneme_chars);
        
        // Create inputs
        let (input_ids, input_lengths, scales) = self.create_input(
            ids,
            inference_cfg.length_scale,
            inference_cfg.noise_w,
            inference_cfg.noise_scale,
        );

        // Run inference
        let input_value = Value::from_array(input_ids)?;
        let lengths_value = Value::from_array(input_lengths)?;
        let scales_value = Value::from_array(scales)?;
        
        let outputs = self.session.run(inputs![
            "input" => &input_value,
            "input_lengths" => &lengths_value,
            "scales" => &scales_value,
        ])?;

        // Extract audio samples
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
        
        // Add EOS
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