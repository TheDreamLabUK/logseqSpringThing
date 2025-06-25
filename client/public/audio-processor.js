/**
 * Audio Worklet Processor for real-time audio processing
 * Handles audio level detection and optional processing
 */

class AudioProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.bufferSize = options.processorOptions.bufferSize || 4096;
    this.sampleBuffer = [];
    this.lastUpdateTime = 0;
    this.updateInterval = 100; // ms
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];

    if (!input || !input[0]) {
      return true;
    }

    const inputChannel = input[0];
    const outputChannel = output[0];

    // Copy input to output (passthrough)
    if (outputChannel) {
      outputChannel.set(inputChannel);
    }

    // Calculate RMS for audio level
    let sum = 0;
    for (let i = 0; i < inputChannel.length; i++) {
      sum += inputChannel[i] * inputChannel[i];
    }
    const rms = Math.sqrt(sum / inputChannel.length);
    const audioLevel = Math.min(1, rms * 5); // Normalize and scale

    // Send audio level updates at interval
    const currentTime = Date.now();
    if (currentTime - this.lastUpdateTime > this.updateInterval) {
      this.port.postMessage({
        type: 'audioLevel',
        level: audioLevel,
        timestamp: currentTime
      });
      this.lastUpdateTime = currentTime;
    }

    // Buffer samples if needed for future processing
    this.sampleBuffer.push(...inputChannel);
    if (this.sampleBuffer.length > this.bufferSize) {
      this.sampleBuffer = this.sampleBuffer.slice(-this.bufferSize);
    }

    return true;
  }
}

registerProcessor('audio-processor', AudioProcessor);