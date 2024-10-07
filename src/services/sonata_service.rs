use std::path::Path;
use sonata_synth::SonataSpeechSynthesizer;
use sonata_piper::from_config_path;
use anyhow::Result;

pub struct SonataService {
    synthesizer: SonataSpeechSynthesizer,
}

impl SonataService {
    pub fn new(voice_config_path: &str) -> Result<Self> {
        let voice = from_config_path(Path::new(voice_config_path))?;
        let synthesizer = SonataSpeechSynthesizer::new(voice)?;
        Ok(SonataService { synthesizer })
    }

    pub async fn synthesize(&self, text: &str) -> Result<Vec<u8>> {
        // Assuming the synthesize method returns a Vec<u8> of audio data
        self.synthesizer.synthesize(text).map_err(Into::into)
    }
}