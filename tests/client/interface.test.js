import { Interface } from '../../public/js/components/interface';

describe('uInterface', () => {
  let uInterface;
  let mockDocument;

  beforeEach(() => {
    mockDocument = {
      createElement: jest.fn().mockReturnValue({
        style: {},
        appendChild: jest.fn(),
        addEventListener: jest.fn()
      }),
      body: {
        appendChild: jest.fn(),
        removeChild: jest.fn()
      },
      getElementById: jest.fn().mockReturnValue({
        style: {},
        innerHTML: '',
        appendChild: jest.fn()
      })
    };
    uInterface = new Interface(mockDocument);
  });

  test('Interface initializes correctly', () => {
    expect(uInterface.document).toBe(mockDocument);
  });

  test('createUI creates info panel', () => {
    uInterface.createUI();

    expect(mockDocument.createElement).toHaveBeenCalledWith('div');
    expect(mockDocument.body.appendChild).toHaveBeenCalled();
  });

  test('updateNodeInfoUI updates info panel with node data', () => {
    const mockNode = { id: 1, name: 'Test Node', size: 10 };
    uInterface.updateNodeInfoUI(mockNode);

    const infoPanel = mockDocument.getElementById('node-info-panel');
    expect(infoPanel.innerHTML).toContain('Test Node');
    expect(infoPanel.style.display).toBe('block');
  });

  test('updateNodeInfoUI hides info panel when no node is selected', () => {
    uInterface.updateNodeInfoUI(null);

    const infoPanel = mockDocument.getElementById('node-info-panel');
    expect(infoPanel.style.display).toBe('none');
  });

  test('createChatInterface creates chat container and elements', () => {
    uInterface.createChatInterface();

    expect(mockDocument.createElement).toHaveBeenCalledWith('div');
    expect(mockDocument.createElement).toHaveBeenCalledWith('input');
    expect(mockDocument.body.appendChild).toHaveBeenCalled();
  });

  test('addChatMessage adds message to chat container', () => {
    const mockChatMessages = {
      appendChild: jest.fn(),
      scrollTop: 0,
      scrollHeight: 100
    };
    mockDocument.getElementById.mockReturnValue(mockChatMessages);
    uInterface.createChatInterface();
    uInterface.addChatMessage('User', 'Test message');

    expect(mockDocument.createElement).toHaveBeenCalledWith('p');
    expect(mockChatMessages.appendChild).toHaveBeenCalled();
    expect(mockChatMessages.scrollTop).toBe(mockChatMessages.scrollHeight);
  });

  test('displayErrorMessage creates and removes error container', () => {
    jest.useFakeTimers();
    uInterface.displayErrorMessage('Test error');

    expect(mockDocument.createElement).toHaveBeenCalledWith('div');
    expect(mockDocument.body.appendChild).toHaveBeenCalled();

    jest.runAllTimers();

    expect(mockDocument.body.removeChild).toHaveBeenCalled();
  });
});
