import { Matrix } from "@oxide-js/core";
import { BaseLayer } from "@oxide-js/layers";
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

  public forward(inputs: Matrix, isTraining: boolean = this.training): Matrix {
    this.assertNotEmpty();

    if (!this.isBuilt) {
      this.build(inputs._shape);
    }

    let output = inputs;

    for (const layer of this.layers) {
      output = layer.forward(output, isTraining);
    }

    this.outputShape = [...output._shape];

    return output;
  }
}
