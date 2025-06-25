import React, { useState, useEffect } from 'react';
import { Mic, MicOff, Volume2 } from 'lucide-react';
import { VoiceWebSocketService } from '../services/VoiceWebSocketService';
import { useSettingsStore } from '../store/settingsStore';

export interface VoiceButtonProps {
  className?: string;
  size?: 'sm' | 'md' | 'lg';
  variant?: 'primary' | 'secondary' | 'ghost';
}

export const VoiceButton: React.FC<VoiceButtonProps> = ({
  className = '',
  size = 'md',
  variant = 'primary'
}) => {
  const [isListening, setIsListening] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  const settings = useSettingsStore((state) => state.settings);
  const voiceService = VoiceWebSocketService.getInstance();

  useEffect(() => {
    // Set up event listeners
    const handleAudioLevel = (level: number) => setAudioLevel(level);
    const handleStateChange = (state: string) => {
      setIsPlaying(state === 'playing');
    };

    const audioInput = voiceService.getAudioInput();
    const audioOutput = voiceService.getAudioOutput();

    audioInput.on('audioLevel', handleAudioLevel);
    audioOutput.on('stateChange', handleStateChange);

    return () => {
      audioInput.off('audioLevel', handleAudioLevel);
      audioOutput.off('stateChange', handleStateChange);
    };
  }, []);

  const toggleListening = async () => {
    try {
      if (isListening) {
        voiceService.stopAudioStreaming();
        setIsListening(false);
        setAudioLevel(0);
      } else {
        await voiceService.startAudioStreaming();
        setIsListening(true);
      }
    } catch (error) {
      console.error('Failed to toggle voice input:', error);
    }
  };

  const sizeClasses = {
    sm: 'h-8 w-8',
    md: 'h-10 w-10',
    lg: 'h-12 w-12'
  };

  const variantClasses = {
    primary: 'bg-blue-500 hover:bg-blue-600 text-white',
    secondary: 'bg-gray-200 hover:bg-gray-300 text-gray-700',
    ghost: 'hover:bg-gray-100 text-gray-600'
  };

  const buttonClasses = `
    ${sizeClasses[size]}
    ${variantClasses[variant]}
    ${className}
    relative flex items-center justify-center
    rounded-full transition-all duration-200
    focus:outline-none focus:ring-2 focus:ring-blue-400
    ${isListening ? 'ring-2 ring-red-400' : ''}
  `;

  return (
    <button
      onClick={toggleListening}
      className={buttonClasses}
      aria-label={isListening ? 'Stop voice input' : 'Start voice input'}
    >
      {/* Audio level indicator */}
      {isListening && (
        <div
          className="absolute inset-0 rounded-full bg-red-400 opacity-30 animate-pulse"
          style={{
            transform: `scale(${1 + audioLevel * 0.5})`,
            transition: 'transform 100ms ease-out'
          }}
        />
      )}

      {/* Icon */}
      {isPlaying ? (
        <Volume2 className="w-5 h-5" />
      ) : isListening ? (
        <Mic className="w-5 h-5" />
      ) : (
        <MicOff className="w-5 h-5" />
      )}
    </button>
  );
};