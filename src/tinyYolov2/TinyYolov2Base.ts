import * as tf from '@tensorflow/tfjs-core'

import { BoundingBox } from '../classes/BoundingBox'
import { Dimensions } from '../classes/Dimensions'
import { ObjectDetection } from '../classes/ObjectDetection'
import { convLayer } from '../common'
import { ConvParams, SeparableConvParams } from '../common/types'
import { toNetInput } from '../dom'
import { NetInput } from '../dom/NetInput'
import { TNetInput } from '../dom/types'
import { NeuralNetwork } from '../NeuralNetwork'
import { sigmoid } from '../ops'
import { nonMaxSuppression } from '../ops/nonMaxSuppression'
import { normalize } from '../ops/normalize'
import { TinyYolov2Config, validateConfig } from './config'
import { convWithBatchNorm } from './convWithBatchNorm'
import { depthwiseSeparableConv } from './depthwiseSeparableConv'
import { extractParams } from './extractParams'
import { extractParamsFromWeigthMap } from './extractParamsFromWeigthMap'
import { leaky } from './leaky'
import { ITinyYolov2Options, TinyYolov2Options } from './TinyYolov2Options'
import {
  DefaultTinyYolov2NetParams,
  MobilenetParams,
  TinyYolov2NetParams
} from './types'
import { features } from 'process'

export class TinyYolov2Base extends NeuralNetwork<TinyYolov2NetParams> {
  public static DEFAULT_FILTER_SIZES = [
    3,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    1024
  ]

  save_conv1: any
  save_conv4: any
  save_conv7: any
  param0: any
  param3: any

  private _config: TinyYolov2Config

  constructor(config: TinyYolov2Config) {
    super('TinyYolov2')
    validateConfig(config)
    this._config = config
  }

  public get config(): TinyYolov2Config {
    return this._config
  }

  public get withClassScores(): boolean {
    return this.config.withClassScores || this.config.classes.length > 1
  }

  public get boxEncodingSize(): number {
    return 5 + (this.withClassScores ? this.config.classes.length : 0)
  }

  public runTinyYolov2(x: tf.Tensor4D, params: DefaultTinyYolov2NetParams) {
    let out = convWithBatchNorm(x, params.conv0)
    out = tf.maxPool(out, [2, 2], [2, 2], 'same')
    let get_conv1 = out
    out = convWithBatchNorm(out, params.conv1)
    out = tf.maxPool(out, [2, 2], [2, 2], 'same')
    out = convWithBatchNorm(out, params.conv2)
    out = tf.maxPool(out, [2, 2], [2, 2], 'same')
    out = convWithBatchNorm(out, params.conv3)
    out = tf.maxPool(out, [2, 2], [2, 2], 'same')
    let get_conv4 = out
    out = convWithBatchNorm(out, params.conv4)
    out = tf.maxPool(out, [2, 2], [2, 2], 'same')
    out = convWithBatchNorm(out, params.conv5)
    out = tf.maxPool(out, [2, 2], [1, 1], 'same')
    out = convWithBatchNorm(out, params.conv6)
    out = convWithBatchNorm(out, params.conv7)
    out = convLayer(out, params.conv8, 'valid', false)
    let get_conv7 = out
    return {
      out,
      save_conv1: get_conv1,
      save_conv4: get_conv4,
      save_conv7: get_conv7,
      param0: params.conv0.conv.filters,
      param3: params.conv3.conv.filters
    }
  }

  public runMobilenet(x: tf.Tensor4D, params: MobilenetParams) {
    let out = this.config.isFirstLayerConv2d
      ? leaky(convLayer(x, params.conv0 as ConvParams, 'valid', false))
      : depthwiseSeparableConv(x, params.conv0 as SeparableConvParams)
    out = tf.maxPool(out, [2, 2], [2, 2], 'same')
    let get_conv1 = out
    out = depthwiseSeparableConv(out, params.conv1)
    out = tf.maxPool(out, [2, 2], [2, 2], 'same')
    out = depthwiseSeparableConv(out, params.conv2)
    out = tf.maxPool(out, [2, 2], [2, 2], 'same')
    out = depthwiseSeparableConv(out, params.conv3)
    out = tf.maxPool(out, [2, 2], [2, 2], 'same')
    let get_conv4 = out
    out = depthwiseSeparableConv(out, params.conv4)
    out = tf.maxPool(out, [2, 2], [2, 2], 'same')
    out = depthwiseSeparableConv(out, params.conv5)
    out = tf.maxPool(out, [2, 2], [1, 1], 'same')
    out = params.conv6 ? depthwiseSeparableConv(out, params.conv6) : out
    out = params.conv7 ? depthwiseSeparableConv(out, params.conv7) : out
    out = convLayer(out, params.conv8, 'valid', false)
    let get_conv7 = out
    let param = params.conv0 as ConvParams
    return {
      out,
      save_conv1: get_conv1,
      save_conv4: get_conv4,
      save_conv7: get_conv7,
      param0: param.filters,
      param3: params.conv3.depthwise_filter
    }
  }

  public forwardInput(input: NetInput, inputSize: number): tf.Tensor4D {
    const { params } = this

    if (!params) {
      throw new Error('TinyYolov2 - load model before inference')
    }

    return tf.tidy(() => {
      let batchTensor = input.toBatchTensor(inputSize, false).toFloat()
      batchTensor = this.config.meanRgb
        ? normalize(batchTensor, this.config.meanRgb)
        : batchTensor
      batchTensor = batchTensor.div(tf.scalar(256)) as tf.Tensor4D

      const features = this.config.withSeparableConvs
        ? this.runMobilenet(batchTensor, params as MobilenetParams)
        : this.runTinyYolov2(batchTensor, params as DefaultTinyYolov2NetParams)
      this.save_conv1 = tf.transpose(features.save_conv1, [0, 3, 1, 2]).reshape([16, 111, 111]).arraySync()
      this.save_conv4 = tf.transpose(features.save_conv4, [0, 3, 1, 2]).reshape([128, 14, 14]).arraySync()
      this.save_conv7 = tf.transpose(features.save_conv7.sub(features.save_conv7.min()).div(features.save_conv7.max().sub(features.save_conv7.min())).mul(255.0), [0, 3, 1, 2]).arraySync()
      this.param0 = tf.transpose(features.param0, [3, 2, 0, 1]).arraySync()
      this.param3 = tf.transpose(features.param3, [3, 2, 0, 1]).arraySync()
      return features.out
    })
  }

  public async forward(
    input: TNetInput,
    inputSize: number
  ): Promise<tf.Tensor4D> {
    return await this.forwardInput(await toNetInput(input), inputSize)
  }

  public async getConv1() {
    return this.save_conv1
  }

  public async getConv4() {
    return this.save_conv4
  }

  public async getParam0() {
    return this.param0
  }

  public async getParam3() {
    return this.param3
  }

  public async getConvLayerString() {
    return this.save_conv4.toString()
  }

  public async getKernel_0(list: number[]) {
    return tf.tidy(() => {
      // const list = [2, 8, 11]
      var grayScale = []
      for (let i = 0; i < 3; i++) {
        var saveconv = this.param0.slice(list[i], list[i] + 1)[0]
        var maxRow = saveconv[0].map(function (row: number[]) {
          return Math.max.apply(Math, row.map(Math.abs))
        })
        maxRow = maxRow.concat(
          saveconv[1].map(function (row: number[]) {
            return Math.max.apply(Math, row.map(Math.abs))
          })
        )
        maxRow = maxRow.concat(
          saveconv[2].map(function (row: number[]) {
            return Math.max.apply(Math, row.map(Math.abs))
          })
        )
        var max = Math.max.apply(null, maxRow)
        for (var j = 0; j < 3; j++) {
          var saveconv = this.param0.slice(list[i], list[i] + 1)[0][j]
          saveconv = saveconv.map(function (x: any[]) {
            return x.map(function (y) {
              return y * 255 / max
            })
          })
          var one_minus_saveconv = saveconv.map(function (x: any[]) {
            return x.map(function (y) {
              return Math.max(0, -y)
            })
          })
          saveconv = saveconv.map(function (x: any[]) {
            return x.map(function (y) {
              return Math.max(0, y)
            })
          })
          const alpha = tf.fill([3, 3], 255)
          var grayScaleImage
          if (j == 0) {
            grayScaleImage = tf.stack(
              [saveconv, one_minus_saveconv, one_minus_saveconv, alpha],
              2
            )
          } else if (j == 1) {
            grayScaleImage = tf.stack(
              [one_minus_saveconv, saveconv, one_minus_saveconv, alpha],
              2
            )
          } else {
            grayScaleImage = tf.stack(
              [one_minus_saveconv, one_minus_saveconv, saveconv, alpha],
              2
            )
          }
          grayScale.push(grayScaleImage.as1D().arraySync())
        }
      }
      return grayScale
    })
  }

  public async getGrayScale(list: number[]) {
    return tf.tidy(() => {
      // const list = [2, 8, 11]
      var grayScale = []
      for (let i = 0; i < 3; i++) {
        let saveconv = this.save_conv1.slice(list[i], list[i] + 1)[0]
        var maxRow = saveconv.map(function (row: any) {
          return Math.max.apply(Math, row)
        })
        var max = Math.max.apply(null, maxRow)
        var minRow = saveconv.map(function (row: any) {
          return Math.min.apply(Math, row)
        })
        var min = Math.min.apply(null, minRow)
        saveconv = saveconv.map(function (x: any[]) {
          return x.map(function (y) {
            return ((y - min) * 255) / (max - min)
          })
        })
        const alpha = tf.fill([111, 111], 255)
        const grayScaleImage = tf.stack(
          [saveconv, saveconv, saveconv, alpha],
          2
        )
        grayScale.push(grayScaleImage.as1D().arraySync())
      }
      return grayScale
    })
  }

  public async getGrayScale_conv4(list: number[]) {
    return tf.tidy(() => {
      var grayScale = []
      for (let i = 0; i < 4; i++) {
        let saveconv = this.save_conv4.slice(list[i], list[i] + 1)[0]
        var maxRow = saveconv.map(function (row: any) {
          return Math.max.apply(Math, row)
        })
        var max = Math.max.apply(null, maxRow)
        var minRow = saveconv.map(function (row: any) {
          return Math.min.apply(Math, row)
        })
        var min = Math.min.apply(null, minRow)
        saveconv = saveconv.map(function (x: any[]) {
          return x.map(function (y) {
            return ((y - min) * 255) / (max - min)
          })
        })
        const alpha = tf.fill([14, 14], 255)
        const grayScaleImage = tf.stack([saveconv, saveconv, saveconv, alpha], 2)
        grayScale.push(grayScaleImage.as1D().arraySync())
      }
      return grayScale
    })
  }

  public async detect(
    input: TNetInput,
    forwardParams: ITinyYolov2Options = {}
  ): Promise<ObjectDetection[]> {
    const { inputSize, scoreThreshold } = new TinyYolov2Options(forwardParams)

    const netInput = await toNetInput(input)
    const out = await this.forwardInput(netInput, inputSize)
    const out0 = tf.tidy(() => tf.unstack(out)[0].expandDims()) as tf.Tensor4D

    const inputDimensions = {
      width: netInput.getInputWidth(0),
      height: netInput.getInputHeight(0)
    }

    const results = await this.extractBoxes(
      out0,
      netInput.getReshapedInputDimensions(0),
      scoreThreshold
    )
    out.dispose()
    out0.dispose()

    const boxes = results.map(res => res.box)
    const scores = results.map(res => res.score)
    const classScores = results.map(res => res.classScore)
    const classNames = results.map(res => this.config.classes[res.label])

    const indices = nonMaxSuppression(
      boxes.map(box => box.rescale(inputSize)),
      scores,
      this.config.iouThreshold,
      true
    )

    const detections = indices.map(
      idx =>
        new ObjectDetection(
          scores[idx],
          classScores[idx],
          classNames[idx],
          boxes[idx],
          inputDimensions
        )
    )

    return detections
  }

  protected getDefaultModelName(): string {
    return ''
  }

  protected extractParamsFromWeigthMap(weightMap: tf.NamedTensorMap) {
    return extractParamsFromWeigthMap(weightMap, this.config)
  }

  protected extractParams(weights: Float32Array) {
    const filterSizes =
      this.config.filterSizes || TinyYolov2Base.DEFAULT_FILTER_SIZES

    const numFilters = filterSizes ? filterSizes.length : undefined
    if (numFilters !== 7 && numFilters !== 8 && numFilters !== 9) {
      throw new Error(
        `TinyYolov2 - expected 7 | 8 | 9 convolutional filters, but found ${numFilters} filterSizes in config`
      )
    }
    return extractParams(
      weights,
      this.config,
      this.boxEncodingSize,
      filterSizes
    )
  }

  protected async extractBoxes(
    outputTensor: tf.Tensor4D,
    inputBlobDimensions: Dimensions,
    scoreThreshold?: number
  ) {
    const { width, height } = inputBlobDimensions
    const inputSize = Math.min(width, height)
    const correctionFactorX = inputSize / inputSize
    const correctionFactorY = inputSize / inputSize

    const numCells = outputTensor.shape[1]
    const numBoxes = this.config.anchors.length

    const [boxesTensor, scoresTensor, classScoresTensor] = tf.tidy(() => {
      const reshaped = outputTensor.reshape([numCells, numCells, numBoxes, this.boxEncodingSize])

      const boxes = reshaped.slice(
        [0, 0, 0, 0],
        [numCells, numCells, numBoxes, 4]
      )
      const scores = reshaped.slice(
        [0, 0, 0, 4],
        [numCells, numCells, numBoxes, 1]
      )
      const classScores = this.withClassScores
        ? tf.softmax(
          reshaped.slice(
            [0, 0, 0, 5],
            [numCells, numCells, numBoxes, this.config.classes.length]
          ),
          3
        )
        : tf.scalar(0)
      return [boxes, scores, classScores]
    })

    const results = []

    const scoresData = await scoresTensor.array()
    const boxesData = await boxesTensor.array()
    for (let row = 0; row < numCells; row++) {
      for (let col = 0; col < numCells; col++) {
        for (let anchor = 0; anchor < numBoxes; anchor++) {
          const score = sigmoid(scoresData[row][col][anchor][0])
          if (!scoreThreshold || score > scoreThreshold) {
            const ctX = ((col + sigmoid(boxesData[row][col][anchor][0])) / numCells) * correctionFactorX
            const ctY = ((row + sigmoid(boxesData[row][col][anchor][1])) / numCells) *
              correctionFactorY
            const width =
              ((Math.exp(boxesData[row][col][anchor][2]) *
                this.config.anchors[anchor].x) /
                numCells) *
              correctionFactorX
            const height =
              ((Math.exp(boxesData[row][col][anchor][3]) *
                this.config.anchors[anchor].y) /
                numCells) *
              correctionFactorY

            const x = ctX - width / 2
            const y = ctY - height / 2

            const pos = { row, col, anchor }
            const { classScore, label } = this.withClassScores
              ? await this.extractPredictedClass(
                classScoresTensor as tf.Tensor4D,
                pos
              )
              : { classScore: 1, label: 0 }

            results.push({
              box: new BoundingBox(x, y, x + width, y + height),
              score: score,
              classScore: score * classScore,
              label,
              ...pos
            })
          }
        }
      }
    }

    boxesTensor.dispose()
    scoresTensor.dispose()
    classScoresTensor.dispose()

    return results
  }

  private async extractPredictedClass(
    classesTensor: tf.Tensor4D,
    pos: { row: number; col: number; anchor: number }
  ) {
    const { row, col, anchor } = pos
    const classesData = await classesTensor.array()
    return Array(this.config.classes.length)
      .fill(0)
      .map((_, i) => classesData[row][col][anchor][i])
      .map((classScore, label) => ({
        classScore,
        label
      }))
      .reduce((max, curr) => (max.classScore > curr.classScore ? max : curr))
  }
}
