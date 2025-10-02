pub mod g2p;
pub mod tts;

use anyhow::Result;
use tts::Tts;
use g2p::G2p;

pub struct PhonikudTts {
    tts: Tts,
    g2p: G2p,
}

impl PhonikudTts {
    pub fn new(tts_model_path: String, tts_config_path: String, g2p_model_path: String) -> Result<Self> {
        let tts = Tts::new(tts_model_path, tts_config_path)?;
        let g2p = G2p::new(g2p_model_path);
        Ok(Self { tts, g2p })
    }

    pub fn create(&mut self, text: String, _audio_path: String, is_phonemes: bool) -> Result<(Vec<f32>, i64)> {
        let phonemes = if is_phonemes {
            text
        } else {
            self.g2p.g2p(&text)
        };
        
        self.tts.create(phonemes)
    }
}

/// Save audio samples to a WAV file
pub fn save_wav(samples: &[f32], sample_rate: i64, path: &str) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sample_rate as u32,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(path, spec)?;
    
    for &sample in samples {
        let amplitude = (sample * i16::MAX as f32) as i16;
        writer.write_sample(amplitude)?;
    }
    
    writer.finalize()?;
    Ok(())
}