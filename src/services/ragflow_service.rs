use crate::handlers::ragflow_handler::Message;

pub struct RAGFlowService;

impl RAGFlowService {
    pub async fn create_conversation(user_id: String) -> Result<String, reqwest::Error> {
        // Implementation goes here
        Ok(String::new())
    }

    pub async fn send_message(conversation_id: String, message: String) -> Result<String, reqwest::Error> {
        // Implementation goes here
        Ok(String::new())
    }

    pub async fn get_chat_history(conversation_id: String) -> Result<Vec<Message>, reqwest::Error> {
        // Implementation goes here
        Ok(Vec::new())
    }
}
