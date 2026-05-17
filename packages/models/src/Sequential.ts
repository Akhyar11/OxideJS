import { Matrix } from "@oxide-js/core";
import { BaseLayer, ForwardOptions } from "@oxide-js/layers";
import { BaseModel } from "./BaseModel.js";
import { ModelConfig } from "./types.js";

export class Sequential extends BaseModel {
  constructor(layers: BaseLayer[] = [], config?: ModelConfig) {
    super(config);

    for (const layer of layers) {
      this.add(layer);
    }
  }

  public override add(layer: BaseLayer): this {
    this.layers.push(layer);
    this.makeUniqueLayerName(layer);
    this.isBuilt = false;
    return this;
  }

  public forward(inputs: Matrix, optionsOrTraining: ForwardOptions | boolean = this.training): Matrix {
    this.assertNotEmpty();

    // Convert boolean to ForwardOptions for compatibility
    const options: ForwardOptions =
      typeof optionsOrTraining === "boolean"
        ? { training: optionsOrTraining }
        : optionsOrTraining;

    if (!this.isBuilt) {
      this.build(inputs._shape);
    }

    let output = inputs;

    for (const layer of this.layers) {
      output = layer.forward(output, options);
    }

    this.outputShape = [...output._shape];

    return output;
  }
}
