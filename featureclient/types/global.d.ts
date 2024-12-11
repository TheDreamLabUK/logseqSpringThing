declare global {
  interface Window {
    __VUE_PROD_DEVTOOLS__?: boolean;
    __VUE_PROD_ERROR_HANDLER__?: (err: Error, vm: any, info: string) => void;
  }
}

export {};
