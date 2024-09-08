export function setupWebSocket(url) {
  const ws = new WebSocket(url);
  ws.onopen = () => console.log('WebSocket connection established');
  ws.onmessage = (event) => console.log('Received:', event.data);
  return ws;
}
