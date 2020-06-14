import * as tf from '@tensorflow/tfjs-core';

import { Rect } from '../classes';
import { FaceDetection } from '../classes/FaceDetection';
import { NetInput, TNetInput, toNetInput } from '../dom';
import { NeuralNetwork } from '../NeuralNetwork';
import { extractParams } from './extractParams';
import { extractParamsFromWeigthMap } from './extractParamsFromWeigthMap';
import { mobileNetV1 } from './mobileNetV1';
import { nonMaxSuppression } from './nonMaxSuppression';
import { outputLayer } from './outputLayer';
import { predictionLayer } from './predictionLayer';
import { ISsdMobilenetv1Options, SsdMobilenetv1Options } from './SsdMobilenetv1Options';
import { NetParams } from './types';
import { tensor2d } from '@tensorflow/tfjs-core';


export class SsdMobilenetv1 extends NeuralNetwork<NetParams> {

  constructor() {
    super('SsdMobilenetv1')
  }
  input_grayScale: number | number[] | number[][] | number[][][] | number[][][][] | number[][][][][] | number[][][][][][];
  save_conv1: any;
  save_conv11: any;

  public forwardInput(input: NetInput) {

    const { params } = this

    if (!params) {
      throw new Error('SsdMobilenetv1 - load model before inference')
    }

    return tf.tidy(() => {
      tf.dispose(this.save_conv1)
      const batchTensor = input.toBatchTensor(512, false).toFloat()

      const x = tf.sub(tf.mul(batchTensor, tf.scalar(0.007843137718737125)), tf.scalar(1)) as tf.Tensor4D
      const features = mobileNetV1(x, params.mobilenetv1)
      this.input_grayScale = batchTensor.reshape([512, 512, 3]).arraySync();
      this.save_conv1 = tf.transpose(features.save_conv1, [0, 3, 1, 2]).reshape([64, 256, 256]).arraySync();
      this.save_conv11 = tf.transpose(features.conv11.mul(-255 / 6.0).add(255), [0, 3, 1, 2]).reshape([512, 32, 32]).arraySync();
      const {
        boxPredictions,
        classPredictions
      } = predictionLayer(features.out, features.conv11, params.prediction_layer)

      return outputLayer(boxPredictions, classPredictions, params.output_layer);
    })
  }

  public async forward(input: TNetInput) {
    return this.forwardInput(await toNetInput(input))
  }

  public async getConvLayerString() {
    return this.save_conv11.toString()
  }

  public async getInputString() {
    return this.input_grayScale.toString()
  }

  public async getConvLayer() {
    return this.save_conv11
  }

  public async getGrayScale() {
    return tf.tidy(() => {
      const list = [42, 26, 54, 55, 60]
      var grayScale = []
      for (let i = 0; i < 5; i++) {
        let saveconv = this.save_conv1.slice(list[i], list[i] + 1)[0]
        // const convertedconv = saveconv[0];
        const alpha = tf.fill([256, 256], 255)
        const grayScaleImage = tf.stack([saveconv, saveconv, saveconv, alpha], 2)
        grayScale.push(grayScaleImage.as1D().arraySync())
      }
      return grayScale
    })
  }

  public async getGrayScale_conv11() {
    return tf.tidy(() => {
      const list = [85, 90, 333, 463]
      var grayScale = []
      for (let i = 0; i < 4; i++) {
        let saveconv = this.save_conv11.slice(list[i], list[i] + 1)[0];
        var maxRow = saveconv.map(function (row: any) { return Math.max.apply(Math, row); });
        var max = Math.max.apply(null, maxRow);
        var minRow = saveconv.map(function (row: any) { return Math.min.apply(Math, row); });
        var min = Math.min.apply(null, minRow);
        saveconv = saveconv.map(function (x: number) { return ((x - min) * 255) / (max - min) });
        // const convertedconv = saveconv[0];
        const alpha = tf.fill([32, 32], 255)
        const grayScaleImage = tf.stack([saveconv, saveconv, saveconv, alpha], 2)
        grayScale.push(grayScaleImage.as1D().arraySync())
      }
      return grayScale
    })
  }

  public async locateFaces(
    input: TNetInput,
    options: ISsdMobilenetv1Options = {}
  ): Promise<FaceDetection[]> {

    const { maxResults, minConfidence } = new SsdMobilenetv1Options(options)

    const netInput = await toNetInput(input)

    const {
      boxes: _boxes,
      scores: _scores
    } = this.forwardInput(netInput)


    // TODO batches
    const boxes = _boxes[0]
    const scores = _scores[0]
    for (let i = 1; i < _boxes.length; i++) {
      _boxes[i].dispose()
      _scores[i].dispose()
    }

    // TODO find a better way to filter by minConfidence
    const scoresData = Array.from(await scores.data())

    const iouThreshold = 0.5
    const indices = nonMaxSuppression(
      boxes,
      scoresData,
      maxResults,
      iouThreshold,
      minConfidence
    )

    const reshapedDims = netInput.getReshapedInputDimensions(0)
    const inputSize = netInput.inputSize as number
    const padX = inputSize / inputSize
    const padY = inputSize / inputSize

    const boxesData = boxes.arraySync()
    const results = indices
      .map(idx => {
        const [top, bottom] = [
          Math.max(0, boxesData[idx][0]),
          Math.min(1.0, boxesData[idx][2])
        ].map(val => val * padY)
        const [left, right] = [
          Math.max(0, boxesData[idx][1]),
          Math.min(1.0, boxesData[idx][3])
        ].map(val => val * padX)
        return new FaceDetection(
          scoresData[idx],
          new Rect(
            left,
            top,
            right - left,
            bottom - top
          ),
          {
            height: netInput.getInputHeight(0),
            width: netInput.getInputWidth(0)
          }
        )
      })

    boxes.dispose()
    scores.dispose()

    return results
  }

  protected getDefaultModelName(): string {
    return 'ssd_mobilenetv1_model'
  }

  protected extractParamsFromWeigthMap(weightMap: tf.NamedTensorMap) {
    return extractParamsFromWeigthMap(weightMap)
  }

  protected extractParams(weights: Float32Array) {
    return extractParams(weights)
  }
}