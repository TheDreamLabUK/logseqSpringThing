import React, { useEffect, useState } from 'react';
import { Mic, Volume2, Loader2 } from 'lucide-react';
import { VoiceWebSocketService, TranscriptionResult } from '../services/VoiceWebSocketService';

export interface VoiceIndicatorProps {
  className?: string;
  showTranscription?: boolean;
  showStatus?: boolean;
}

export const VoiceIndicator: React.FC<VoiceIndicatorProps> = ({
  className = '',
  showTranscription = true,
  showStatus = true
}) => {
  const [status, setStatus] = useState<'idle' | 'listening' | 'processing' | 'speaking'>('idle');
  const [transcription, setTranscription] = useState<string>('');
  const [partialTranscription, setPartialTranscription] = useState<string>('');
  const [audioLevel, setAudioLevel] = useState(0);
  const voiceService = VoiceWebSocketService.getInstance();

  useEffect(() => {
    // Set up event listeners
    const handleTranscription = (result: TranscriptionResult) => {
      if (result.isFinal) {
        setTranscription(result.text);
        setPartialTranscription('');
      } else {
        setPartialTranscription(result.text);
      }
    };

    const handleAudioLevel = (level: number) => setAudioLevel(level);

    const handleAudioStreamingStarted = () => setStatus('listening');
    const handleAudioStreamingStopped = () => setStatus('idle');
    const handleAudioStarted = () => setStatus('speaking');
    const handleAudioEnded = () => setStatus('idle');

    voiceService.on('transcription', handleTranscription);
    voiceService.on('audioStreamingStarted', handleAudioStreamingStarted);
    voiceService.on('audioStreamingStopped', handleAudioStreamingStopped);

    const audioInput = voiceService.getAudioInput();
    const audioOutput = voiceService.getAudioOutput();

    audioInput.on('audioLevel', handleAudioLevel);
    audioOutput.on('audioStarted', handleAudioStarted);
    audioOutput.on('audioEnded', handleAudioEnded);

    return () => {
      voiceService.off('transcription', handleTranscription);
      voiceService.off('audioStreamingStarted', handleAudioStreamingStarted);
      voiceService.off('audioStreamingStopped', handleAudioStreamingStopped);
      audioInput.off('audioLevel', handleAudioLevel);
      audioOutput.off('audioStarted', handleAudioStarted);
      audioOutput.off('audioEnded', handleAudioEnded);
    };
  }, []);

  const statusConfig = {
    idle: { icon: null, text: '', color: 'text-gray-400' },
    listening: { icon: Mic, text: 'Listening...', color: 'text-red-500' },
    processing: { icon: Loader2, text: 'Processing...', color: 'text-blue-500' },
    speaking: { icon: Volume2, text: 'Speaking...', color: 'text-green-500' }
  };

  const currentStatus = statusConfig[status];

  if (status === 'idle' && !transcription && !partialTranscription) {
    return null;
  }

  return (
    <div className={`flex flex-col gap-2 p-3 rounded-lg bg-gray-50 ${className}`}>
      {/* Status indicator */}
      {showStatus && currentStatus.icon && (
        <div className={`flex items-center gap-2 ${currentStatus.color}`}>
          <currentStatus.icon
            className={`w-4 h-4 ${status === 'processing' ? 'animate-spin' : ''}`}
          />
          <span className="text-sm font-medium">{currentStatus.text}</span>

          {/* Audio level bars */}
          {status === 'listening' && (
            <div className="flex gap-1 items-center ml-2">
              {[...Array(5)].map((_, i) => (
                <div
                  key={i}
                  className={`w-1 h-3 bg-current transition-all duration-100 ${
                    audioLevel > (i + 1) / 5 ? 'opacity-100' : 'opacity-20'
                  }`}
                  style={{
                    height: audioLevel > (i + 1) / 5 ? '12px' : '6px'
                  }}
                />
              ))}
            </div>
          )}
        </div>
      )}

      {/* Transcription display */}
      {showTranscription && (transcription || partialTranscription) && (
        <div className="text-sm">
          {transcription && (
            <p className="text-gray-800">{transcription}</p>
          )}
          {partialTranscription && (
            <p className="text-gray-500 italic">{partialTranscription}</p>
          )}
        </div>
      )}
    </div>
  );
};