import { initGraph } from '../../public/js/components/graphSimulation';
describe('GraphSimulation', () => {
  test('initGraph should initialize graph simulation correctly', () => {
    // Mock console.log
    console.log = globalThis.jest?.fn() || jest.fn();
    initGraph();
    expect(console.log).toHaveBeenCalledWith('Initializing graph simulation...');
  });
});