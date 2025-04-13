import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../../../../ui/Card';
import { Label } from '../../../../ui/Label';
import { Input } from '../../../../ui/Input';
import { useSettingsStore } from '../../../../store/settingsStore';
import { produce } from 'immer'; // Assuming immer is used in settingsStore

const AIPanel: React.FC = () => {
  const settings = useSettingsStore((state) => state.settings);
  const updateSettings = useSettingsStore((state) => state.updateSettings);

  // Corrected handler using updateSettings with an updater function
  const handleChange = (path: string, value: any) => {
    updateSettings(draft => { // Pass an updater function to updateSettings
      // Basic path setting - needs refinement for nested structures like objects/arrays
      const keys = path.split('.');
      let current: any = draft; // Operate directly on the draft provided by Immer
      try {
        for (let i = 0; i < keys.length - 1; i++) {
          if (current[keys[i]] === undefined) {
             current[keys[i]] = {}; // Create intermediate objects if they don't exist
          }
          current = current[keys[i]];
        }
        current[keys[keys.length - 1]] = value;
      } catch (error) {
        console.error("Failed to update settings:", error, "Path:", path, "Value:", value);
        // Optionally revert or handle the error state
      }
    }); // Pass updater function directly
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>RAGFlow Settings</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4 items-center">
            <Label htmlFor="ragflow-api-key">API Key</Label>
            <Input
              id="ragflow-api-key"
              type="password"
              value={settings?.ragflow?.api_key || ''}
              onChange={(e) => handleChange('ragflow.api_key', e.target.value)}
              placeholder="Enter RAGFlow API Key"
              className="col-span-1"
            />
            <Label htmlFor="ragflow-agent-id">Agent ID</Label>
            <Input
              id="ragflow-agent-id"
              value={settings?.ragflow?.agent_id || ''}
              onChange={(e) => handleChange('ragflow.agent_id', e.target.value)}
               className="col-span-1"
            />
            {/* TODO: Add other RAGFlow settings (api_base_url, timeout, max_retries, chat_id) */}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Perplexity Settings</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4 items-center">
            <Label htmlFor="perplexity-api-key">API Key</Label>
            <Input
              id="perplexity-api-key"
              type="password"
              value={settings?.perplexity?.api_key || ''}
              onChange={(e) => handleChange('perplexity.api_key', e.target.value)}
              placeholder="Enter Perplexity API Key"
               className="col-span-1"
            />
             <Label htmlFor="perplexity-model">Model</Label>
            <Input
              id="perplexity-model"
              value={settings?.perplexity?.model || ''}
              onChange={(e) => handleChange('perplexity.model', e.target.value)}
               className="col-span-1"
            />
            {/* TODO: Add other Perplexity settings (api_url, max_tokens, temp, top_p, penalties, timeout, rate_limit) */}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>OpenAI Settings</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4 items-center">
            <Label htmlFor="openai-api-key">API Key</Label>
            <Input
              id="openai-api-key"
              type="password"
              value={settings?.openai?.api_key || ''}
              onChange={(e) => handleChange('openai.api_key', e.target.value)}
              placeholder="Enter OpenAI API Key"
               className="col-span-1"
            />
             <Label htmlFor="openai-base-url">Base URL</Label>
            <Input
              id="openai-base-url"
              value={settings?.openai?.base_url || ''}
              onChange={(e) => handleChange('openai.base_url', e.target.value)}
               className="col-span-1"
            />
            {/* TODO: Add other OpenAI settings (timeout, rate_limit) */}
          </div>
        </CardContent>
      </Card>

       <Card>
        <CardHeader>
          <CardTitle>Kokoro TTS Settings</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4 items-center">
            <Label htmlFor="kokoro-api-url">API URL</Label>
            <Input
              id="kokoro-api-url"
              value={settings?.kokoro?.api_url || ''}
              onChange={(e) => handleChange('kokoro.api_url', e.target.value)}
               className="col-span-1"
            />
             <Label htmlFor="kokoro-default-voice">Default Voice</Label>
            <Input
              id="kokoro-default-voice"
              value={settings?.kokoro?.default_voice || ''}
              onChange={(e) => handleChange('kokoro.default_voice', e.target.value)}
               className="col-span-1"
            />
            {/* TODO: Add other Kokoro settings (format, speed, timeout, stream, timestamps, sample_rate) */}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default AIPanel;
