use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::AddBos;
use llama_cpp_2::sampling::LlamaSampler;

pub struct G2p {
    backend: LlamaBackend,
    model: LlamaModel,
}

impl G2p {
    pub fn new(g2p_model_path: String) -> Self {
        
        let backend = LlamaBackend::init().expect("Failed to initialize llama backend");
        let params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&backend, g2p_model_path, &params)
            .expect("Failed to load model");
        
        Self { backend, model }
    }

    pub fn g2p(&self, text: &str) -> String {
        let system_message = "Given the following Hebrew sentence, convert it to IPA phonemes.\nInput Format: A Hebrew sentence.\nOutput Format: A string of IPA phonemes.";
        
        let prompt = format!(
            "<start_of_turn>system\n{}<end_of_turn>\n<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n",
            system_message, text
        );

        let ctx_params = LlamaContextParams::default();
        let mut ctx = self.model
            .new_context(&self.backend, ctx_params)
            .expect("Failed to create context");

        let tokens_list = self.model
            .str_to_token(&prompt, AddBos::Always)
            .expect("Failed to tokenize");

        let mut batch = LlamaBatch::new(512, 1);
        let last_index = tokens_list.len() as i32 - 1;
        
        for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
            let is_last = i == last_index;
            batch.add(token, i, &[0], is_last).unwrap();
        }
        
        ctx.decode(&mut batch).expect("Failed to decode");

        let mut n_cur = batch.n_tokens();
        let n_len = 150; // max tokens
        let mut sampler = LlamaSampler::greedy();
        let mut result = String::new();
        let mut decoder = encoding_rs::UTF_8.new_decoder();

        while n_cur <= n_len {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);

            if token == self.model.token_eos() {
                break;
            }

            let output_bytes = self.model.token_to_bytes(token, llama_cpp_2::model::Special::Tokenize).unwrap();
            let mut output_string = String::with_capacity(32);
            let _decode_result = decoder.decode_to_string(&output_bytes, &mut output_string, false);
            
            // Check for stop sequences
            result.push_str(&output_string);
            if result.contains("<end_of_turn>") || result.contains("</s>") {
                break;
            }

            batch.clear();
            batch.add(token, n_cur, &[0], true).unwrap();
            n_cur += 1;
            
            ctx.decode(&mut batch).expect("Failed to decode");
        }

        // Clean up stop sequences from result
        result = result.replace("<end_of_turn>", "");
        result = result.replace("</s>", "");
        result.trim().to_string()
    }
}