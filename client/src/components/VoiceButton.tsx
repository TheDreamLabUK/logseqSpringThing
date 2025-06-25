import React, { useState, useEffect } from 'react';
import { useVoiceInteraction } from '../hooks/useVoiceInteraction';
import { VoiceWebSocketService } from '../services/VoiceWebSocketService';

// Simple icon components to avoid lucide-react import issues
const MicIcon = ({ className }: { className?: string }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
    <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
    <line x1="12" y1="19" x2="12" y2="23" />
    <line x1="8" y1="23" x2="16" y2="23" />
  </svg>
);

const MicOffIcon = ({ className }: { className?: string }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <line x1="1" y1="1" x2="23" y2="23" />
    <path d="M9 9v3a3 3 0 0 0 5.12 2.12L9 9z" />
    <path d="M12 1a3 3 0 0 0-3 3v8.5l6-6V4a3 3 0 0 0-3-3z" />
    <path d="M19 10v2a7 7 0 0 1-.64 3.07L21 17.5A9 9 0 0 0 19 10z" />
    <line x1="12" y1="19" x2="12" y2="23" />
    <line x1="8" y1="23" x2="16" y2="23" />
  </svg>
);

const VolumeIcon = ({ className }: { className?: string }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polygon points="11 5,6 9,2 9,2 15,6 15,11 19,11 5" />
    <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07" />
  </svg>
);

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
  const [audioLevel, setAudioLevel] = useState(0);
  const {
    isListening,
    isSpeaking,
    toggleListening
  } = useVoiceInteraction({
    onError: (error) => console.error('Voice interaction error:', error)
  });

  // Get audio level from VoiceWebSocketService directly for visual feedback
  useEffect(() => {
    const voiceService = VoiceWebSocketService.getInstance();
    const handleAudioLevel = (level: number) => setAudioLevel(level);
    const audioInput = voiceService.getAudioInput();

    audioInput.on('audioLevel', handleAudioLevel);

    return () => {
      audioInput.off('audioLevel', handleAudioLevel);
    };
  }, []);

  const handleToggle = async () => {
    try {
      await toggleListening();
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
      onClick={handleToggle}
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
      {isSpeaking ? (
        <VolumeIcon className="w-5 h-5" />
      ) : isListening ? (
        <MicIcon className="w-5 h-5" />
      ) : (
        <MicOffIcon className="w-5 h-5" />
      )}
    </button>
  );
};