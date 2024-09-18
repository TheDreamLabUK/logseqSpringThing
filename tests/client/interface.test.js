import { Interface } from '../../public/js/components/interface';

describe('uInterface', () => {
  let interface;
  let mockDocument;

  beforeEach(() => {
    mockDocument = {
      createElement: jest.fn().mockReturnValue({
        style: {},
        appendChild: jest.fn(),
        addEventListener: jest.fn()
      }),
      body: {
        appendChild: jest.fn()
      },
      getElementById: jest.fn().mockReturnValue({
        style: {},
        innerHTML: '',
        appendChild: jest.fn()
      })
    };
    interface = new Interface(mockDocument);
  });

  test('Interface initializes correctly', () => {
    expect(interface.document).toBe(mockDocument);
  });

  test('createUI creates info panel', () => {
    interface.createUI();

    expect(mockDocument.createElement).toHaveBeenCalledWith('div');
    expect(mockDocument.body.appendChild).toHaveBeenCalled();
  });

  test('updateNodeInfoUI updates info panel with node data', () => {
    const mockNode = { id: 1, name: 'Test Node', size: 10 };
    interface.updateNodeInfoUI(mockNode);

    const infoPanel = mockDocument.getElementById('node-info-panel');
    expect(infoPanel.innerHTML).toContain('Test Node');
    expect(infoPanel.style.display).toBe('block');
  });

  test('updateNodeInfoUI hides info panel when no node is selected', () => {
    interface.updateNodeInfoUI(null);

    const infoPanel = mockDocument.getElementById('node-info-panel');
    expect(infoPanel.style.display).toBe('none');
  });

  test('createChatInterface creates chat container and elements', () => {
    interface.createChatInterface();

    expect(mockDocument.createElement).toHaveBeenCalledWith('div');
    expect(mockDocument.createElement).toHaveBeenCalledWith('input');
    expect(mockDocument.body.appendChild).toHaveBeenCalled();
  });

  test('addChatMessage adds message to chat container', () => {
    interface.createChatInterface();
    interface.addChatMessage('User', 'Test message');

    const chatMessages = mockDocument.getElementById('chat-messages');
    expect(chatMessages.innerHTML).toContain('User: Test message');
  });

  test('displayErrorMessage creates and removes error container', () => {
    jest.useFakeTimers();
    interface.displayErrorMessage('Test error');

    expect(mockDocument.createElement).toHaveBeenCalledWith('div');
    expect(mockDocument.body.appendChild).toHaveBeenCalled();

    jest.runAllTimers();

    expect(mockDocument.body.removeChild).toHaveBeenCalled();
  });
});
