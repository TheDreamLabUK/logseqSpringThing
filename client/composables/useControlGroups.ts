import { ref, reactive } from 'vue';
import type { ControlGroup, ControlItem } from '@/types/components';

export function useControlGroups() {
  const collapsedGroups = reactive<Record<string, boolean>>({
    appearance: true,
    material: true,
    physics: true,
    bloom: true,
    environment: true,
    fisheye: true
  });

  const toggleGroup = (groupName: string) => {
    collapsedGroups[groupName] = !collapsedGroups[groupName];
  };

  const createControlGroup = (
    name: string,
    label: string,
    controls: ControlItem[]
  ): ControlGroup => ({
    name,
    label,
    collapsed: collapsedGroups[name],
    controls
  });

  const createRangeControl = (
    name: string,
    label: string,
    value: number,
    min: number,
    max: number,
    step: number
  ): ControlItem => ({
    name,
    label,
    value,
    type: 'range',
    min,
    max,
    step
  });

  const createColorControl = (
    name: string,
    label: string,
    value: string
  ): ControlItem => ({
    name,
    label,
    value,
    type: 'color'
  });

  const createCheckboxControl = (
    name: string,
    label: string,
    value: boolean
  ): ControlItem => ({
    name,
    label,
    value,
    type: 'checkbox'
  });

  return {
    collapsedGroups,
    toggleGroup,
    createControlGroup,
    createRangeControl,
    createColorControl,
    createCheckboxControl
  };
}
