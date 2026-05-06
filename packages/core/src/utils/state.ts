class State {
  private value: any;
  constructor(value?: any) {
    this.value = value ? value : undefined;
  }

  setValue(value: any) {
    this.value = value;
  }

  getValue() {
    return this.value;
  }
}

export const useState = new State();
