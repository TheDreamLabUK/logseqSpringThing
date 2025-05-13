// client/src/types/ragflowTypes.ts
export interface RagflowChatRequestPayload {
  question: string;
  sessionId?: string;
}

export interface RagflowChatResponsePayload {
  answer: string;
  sessionId: string;
}