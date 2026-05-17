import { Matrix, engine } from "@oxide-js/core";
import { BaseLayer } from "@oxide-js/layers";
import {
  CompileConfig,
  FitConfig,
  HistoryRecord,
  ModelConfig,
  ModelSummaryRow,
  SerializedModel,
  WeightData
} from "./types.js";

export abstract class BaseModel {
  public name: string;
  public trainable: boolean;
  public training: boolean = true;

  protected layers: BaseLayer[] = [];

  protected optimizer?: any;
  protected lossFn?: any;
  protected metrics: Array<string | any> = [];

  public isCompiled: boolean = false;
  public isBuilt: boolean = false;

  public inputShape: number[] = [];
  public outputShape: number[] = [];

  constructor(config?: ModelConfig) {
    this.name = config?.name || this.constructor.name;
    this.trainable = config?.trainable ?? true;
  }

  /**
   * Forward utama model.
   * Subclass seperti Sequential wajib implement.
   */
  public abstract forward(inputs: Matrix, isTraining?: boolean): Matrix;

  /**
   * Build model berdasarkan input shape.
   * Subclass boleh override jika punya graph non-linear.
   */
  public build(inputShape: number[]): void {
    this.inputShape = [...inputShape];

    let currentShape = [...inputShape];

    for (const layer of this.layers) {
      if (!layer.isBuilt) {
        layer.build(currentShape);
      }

      currentShape = [...layer.outputShape];
    }

    this.outputShape = currentShape;
    this.isBuilt = true;
  }

  /**
   * Tambah layer ke model.
   * Untuk BaseModel dibuat protected/public supaya Sequential bisa pakai langsung.
   */
  public add(layer: BaseLayer): this {
    this.layers.push(layer);
    this.isBuilt = false;
    return this;
  }

  /**
   * Ambil semua layer.
   */
  public getLayers(): BaseLayer[] {
    return [...this.layers];
  }

  /**
   * Ambil layer berdasarkan index.
   */
  public getLayer(index: number): BaseLayer {
    const layer = this.layers[index];

    if (!layer) {
      throw new Error(`[${this.name}] Layer index ${index} tidak ditemukan.`);
    }

    return layer;
  }

  /**
   * Cari layer berdasarkan nama.
   */
  public getLayerByName(name: string): BaseLayer | undefined {
    return this.layers.find(layer => layer.name === name);
  }

  /**
   * Jumlah layer.
   */
  public get layerCount(): number {
    return this.layers.length;
  }

  /**
   * Ubah model dan semua layer menjadi training mode.
   */
  public train(): void {
    this.training = true;

    for (const layer of this.layers) {
      layer.train();
    }
  }

  /**
   * Ubah model dan semua layer menjadi inference/eval mode.
   */
  public eval(): void {
    this.training = false;

    for (const layer of this.layers) {
      layer.eval();
    }
  }

  /**
   * Semua weights dari semua layer.
   */
  public get weights(): Matrix[] {
    return this.layers.flatMap(layer => layer.weights);
  }

  /**
   * Semua trainable weights.
   */
  public get trainableWeights(): Matrix[] {
    if (!this.trainable) return [];

    return this.layers.flatMap(layer => layer.trainableWeights);
  }

  /**
   * Semua non-trainable weights.
   */
  public get nonTrainableWeights(): Matrix[] {
    if (!this.trainable) {
      return this.layers.flatMap(layer => layer.weights);
    }

    return this.layers.flatMap(layer => layer.nonTrainableWeights);
  }

  /**
   * Alias untuk optimizer.
   */
  public getTrainableParameters(): Matrix[] {
    return this.trainableWeights;
  }

  /**
   * Semua parameter, baik trainable maupun non-trainable.
   */
  public getAllParameters(): Matrix[] {
    return this.weights;
  }

  /**
   * Clear gradient semua trainable parameter.
   * Ini tetap diperlukan walaupun gradient ditangani autodiff.
   */
  public zeroGrad(): void {
    for (const param of this.trainableWeights) {
      param.clearGrad();
    }
  }

  /**
   * Alias.
   */
  public clearGradients(): void {
    this.zeroGrad();
  }

  /**
   * Hitung total parameter semua layer.
   */
  public countParams(): number {
    return this.layers.reduce((total, layer) => {
      return total + layer.countParams();
    }, 0);
  }

  /**
   * Hitung total trainable parameter.
   */
  public countTrainableParams(): number {
    return this.trainableWeights.reduce((total, param) => {
      return total + param._shape.reduce((a, b) => a * b, 1);
    }, 0);
  }

  /**
   * Hitung total non-trainable parameter.
   */
  public countNonTrainableParams(): number {
    return this.nonTrainableWeights.reduce((total, param) => {
      return total + param._shape.reduce((a, b) => a * b, 1);
    }, 0);
  }

  /**
   * Compile model.
   * loss dan optimizer bisa object class atau function dari core.
   */
  public compile(config: CompileConfig): void {
    this.optimizer = config.optimizer;
    this.lossFn = config.loss;
    this.metrics = config.metrics ?? [];
    this.isCompiled = true;
  }

  /**
   * Pastikan model sudah compile sebelum training/evaluate.
   */
  protected assertCompiled(): void {
    if (!this.isCompiled) {
      throw new Error(
        `[${this.name}] Model belum di-compile. Panggil model.compile({ optimizer, loss }) terlebih dahulu.`
      );
    }

    if (!this.lossFn) {
      throw new Error(`[${this.name}] Loss function belum didefinisikan.`);
    }

    if (!this.optimizer) {
      throw new Error(`[${this.name}] Optimizer belum didefinisikan.`);
    }
  }

  /**
   * Hitung loss.
   * Dibuat fleksibel karena loss kamu bisa berupa function atau object.
   */
  protected computeLoss(yPred: Matrix, yTrue: Matrix): any {
    if (!this.lossFn) {
      throw new Error(`[${this.name}] Loss function belum tersedia.`);
    }

    if (typeof this.lossFn === "function") {
      return this.lossFn(yPred, yTrue);
    }

    if (typeof this.lossFn.forward === "function") {
      return this.lossFn.forward(yPred, yTrue);
    }

    if (typeof this.lossFn.compute === "function") {
      return this.lossFn.compute(yPred, yTrue);
    }

    throw new Error(`[${this.name}] Format loss function tidak valid.`);
  }

  /**
   * Optimizer step.
   * Dibuat fleksibel agar cocok dengan beberapa bentuk optimizer.
   */
  protected optimizerStep(): void {
    if (!this.optimizer) {
      throw new Error(`[${this.name}] Optimizer belum tersedia.`);
    }

    const params = this.trainableWeights;

    if (typeof this.optimizer.step === "function") {
      this.optimizer.step(params);
      return;
    }

    if (typeof this.optimizer.update === "function") {
      this.optimizer.update(params);
      return;
    }

    throw new Error(`[${this.name}] Optimizer tidak memiliki method step() atau update().`);
  }

  public trainStep(xBatch: Matrix, yBatch: Matrix): { loss: any; yPred: Matrix } {
    this.assertCompiled();
    this.train();

    this.zeroGrad();

    let yPred: Matrix | undefined;
    const tape = engine.grad(() => {
      yPred = this.forward(xBatch, true);
      return this.computeLoss(yPred, yBatch);
    });

    tape.backward(tape.result);

    this.optimizerStep();

    return { loss: tape.result, yPred: yPred! };
  }

  /**
   * Predict/inference.
   */
  public predict(inputs: Matrix): Matrix {
    this.eval();
    return this.forward(inputs, false);
  }

  /**
   * Evaluate sederhana.
   */
  public evaluate(x: Matrix, y: Matrix): { loss: any; yPred: Matrix } {
    if (!this.lossFn) {
      throw new Error(`[${this.name}] Loss function belum tersedia.`);
    }

    this.eval();

    const yPred = this.forward(x, false);
    const loss = this.computeLoss(yPred, y);

    return { loss, yPred };
  }

  /**
   * Fit sederhana.
   * Untuk versi awal, bisa full-batch dulu.
   * DataLoader/batching bisa kamu tambah nanti.
   */
  public fit(x: Matrix, y: Matrix, config: FitConfig = {}): HistoryRecord[] {
    this.assertCompiled();

    const epochs = config.epochs ?? 1;
    const verbose = config.verbose ?? 1;

    const history: HistoryRecord[] = [];

    for (let epoch = 1; epoch <= epochs; epoch++) {
      const result = this.trainStep(x, y);

      const lossValue = this.extractScalar(result.loss);

      const record: HistoryRecord = {
        epoch,
        loss: lossValue
      };

      history.push(record);

      if (verbose) {
        const lossText =
          typeof lossValue === "number" && Number.isFinite(lossValue)
            ? lossValue.toFixed(6)
            : String(lossValue);

        console.log(`Epoch ${epoch}/${epochs} - loss: ${lossText}`);
      }
    }

    return history;
  }

  /**
   * Ambil nilai scalar dari loss.
   * Disesuaikan dengan Matrix core kamu.
   */
  protected extractScalar(value: any): number | undefined {
    if (typeof value === "number") return value;

    if (value instanceof Matrix) {
      return value._data[0];
    }

    if (value?._data?.length > 0) {
      return value._data[0];
    }

    if (typeof value?.item === "function") {
      return value.item();
    }

    if (typeof value?.data === "number") {
      return value.data;
    }

    return undefined;
  }

  /**
   * Ambil semua weight data dari semua layer.
   */
  public getWeights(): WeightData[] {
    return this.layers.flatMap(layer => layer.getWeights());
  }

  /**
   * Set weights ke layer berdasarkan prefix nama layer.
   */
  public setWeights(weightsData: WeightData[]): void {
    for (const layer of this.layers) {
      const layerWeights = weightsData.filter(w => {
        return w.name.startsWith(`${layer.name}/`);
      });

      if (layerWeights.length > 0) {
        layer.setWeights(layerWeights);
      }
    }
  }

  /**
   * Config model.
   */
  public getConfig(): Record<string, any> {
    return {
      name: this.name,
      trainable: this.trainable,
      layers: this.layers.map(layer => layer.getKerasConfig())
    };
  }

  /**
   * Keras-style config.
   */
  public getKerasConfig(): Record<string, any> {
    return {
      class_name: this.constructor.name,
      config: this.getConfig()
    };
  }

  /**
   * Serialize model.
   */
  public serialize(): SerializedModel {
    return {
      class_name: this.constructor.name,
      name: this.name,
      trainable: this.trainable,
      config: this.getConfig(),
      layers: this.layers.map(layer => layer.getKerasConfig()),
      weights: this.getWeights()
    };
  }

  /**
   * Info summary setiap layer.
   */
  public getSummaryInfo(): ModelSummaryRow[] {
    return this.layers.map(layer => layer.getSummaryInfo());
  }

  /**
   * Cetak summary model.
   */
  public summary(): void {
    const rows = this.getSummaryInfo();

    const divider = "=".repeat(80);

    console.log(divider);
    console.log(`Model: ${this.name}`);
    console.log(divider);
    console.log(
      `${"Layer".padEnd(24)} ${"Type".padEnd(22)} ${"Output Shape".padEnd(18)} ${"Param #".padStart(10)}`
    );
    console.log("-".repeat(80));

    for (const row of rows) {
      console.log(
        `${row.name.padEnd(24)} ${row.type.padEnd(22)} ${row.outputShape.padEnd(18)} ${String(row.paramCount).padStart(10)}`
      );
    }

    console.log("-".repeat(80));
    console.log(`Total params: ${this.countParams().toLocaleString()}`);
    console.log(`Trainable params: ${this.countTrainableParams().toLocaleString()}`);
    console.log(`Non-trainable params: ${this.countNonTrainableParams().toLocaleString()}`);
    console.log(divider);
  }

  /**
   * Validasi minimal input.
   */
  protected assertNotEmpty(): void {
    if (this.layers.length === 0) {
      throw new Error(`[${this.name}] Model tidak memiliki layer.`);
    }
  }

  /**
   * Membuat nama layer unik.
   * Sebaiknya dipanggil oleh Sequential.add().
   */
  protected makeUniqueLayerName(layer: BaseLayer): void {
    const existingNames = new Set(
      this.layers
        .filter(l => l !== layer)
        .map(l => l.name)
    );

    if (!existingNames.has(layer.name)) return;

    const baseName = layer.constructor.name;
    let index = 1;
    let candidate = `${baseName}_${index}`;

    while (existingNames.has(candidate)) {
      index++;
      candidate = `${baseName}_${index}`;
    }

    layer.name = candidate;
  }
}
